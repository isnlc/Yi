import path from "node:path";

const targets = await fetch("http://127.0.0.1:9222/json").then(response => response.json());
const target = targets.find(item =>
  item.type === "page" && item.url.startsWith("http://127.0.0.1:8765/")
);
if (!target) {
  throw new Error("未找到打开的项目页面。");
}

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, {once: true});
  socket.addEventListener("error", reject, {once: true});
});

let nextId = 0;
const pending = new Map();
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (!message.id || !pending.has(message.id)) return;
  const {resolve, reject} = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message));
  else resolve(message.result);
});

function command(method, params = {}) {
  const id = ++nextId;
  socket.send(JSON.stringify({id, method, params}));
  return new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
}

const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
await command("Runtime.enable");
await command("DOM.enable");
const documentNode = await command("DOM.getDocument", {depth: -1});

async function setFiles(selector, files) {
  const result = await command("DOM.querySelector", {
    nodeId: documentNode.root.nodeId,
    selector,
  });
  if (!result.nodeId) throw new Error(`未找到文件输入框：${selector}`);
  await command("DOM.setFileInputFiles", {
    nodeId: result.nodeId,
    files: files.map(file => path.resolve(file)),
  });
  await delay(500);
}

await setFiles("#imgFile", ["docs/figures/architecture_latest.png"]);
await setFiles("#docFile", ["README.md"]);

const state = await command("Runtime.evaluate", {
  expression: `(() => {
    const preview = addMsg("user", "请分析这些附件。", "media");
    appendSentAttachments(preview, pendingAttachments);
    const value = {
      count: pendingAttachments.length,
      kinds: pendingAttachments.map(item => item.kind),
      names: pendingAttachments.map(item => item.file.name),
      draftVisible: !document.getElementById("attachmentDraft").hidden,
      renderedItems: document.querySelectorAll(".attachment-item").length,
      sentFileMark: preview.querySelector(".sent-file .file-mark")?.textContent,
      hasStandaloneText: [...preview.querySelector(".bub").childNodes]
        .some(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim()),
      sentPrompt: preview.querySelector(".bub")?.childNodes[0]?.textContent?.trim() || ""
    };
    preview.remove();
    return value;
  })()`,
  returnByValue: true,
});
await command("Runtime.evaluate", {expression: "clearAttachments()"});

const memoryState = await command("Runtime.evaluate", {
  expression: `(async () => {
    go("mem");
    await loadMemories();
    const item = memItems.find(memory =>
      memory.detail.includes("JailbreakScope") && memory.detail.includes("第 2 页")
    ) || memItems.find(memory => parseDocumentMemory(memory.detail));
    if (!item) return {found: false};
    await openMemory(item.id);
    const detailCard = document.querySelector("#memDetail .doc-memory");
    const initiallyCollapsed = detailCard ? !detailCard.open : false;
    detailCard?.querySelector("summary")?.click();
    const expanded = detailCard?.open || false;
    const contentVisible = Boolean(
      detailCard?.querySelector(".doc-content")?.textContent.trim()
    );
    await openDay(item.date);
    const dayCard = document.querySelector("#sheet .doc-memory");
    const dayCollapsed = dayCard ? !dayCard.open : false;
    closeSheet();
    return {
      found: true,
      filename: parseDocumentMemory(item.detail)?.name,
      initiallyCollapsed,
      expanded,
      contentVisible,
      dayCard: Boolean(dayCard),
      dayCollapsed
    };
  })()`,
  awaitPromise: true,
  returnByValue: true,
});

const markdownState = await command("Runtime.evaluate", {
  expression: `(async () => {
    go("mem");
    await loadMemories();
    const item = memItems.find(memory => memory.detail.includes("【文件《讲稿.md》】"));
    if (!item) return {found: false};
    await openMemory(item.id);
    const card = document.querySelector("#memDetail .doc-memory");
    const parsed = parseDocumentMemory(
      (await (await fetch("/api/memory/" + item.id)).json()).source.text
    );
    return {
      found: true,
      filename: parsed?.name,
      section: parsed?.section,
      collapsed: card ? !card.open : false,
      contentLength: card?.querySelector(".doc-content")?.textContent.length || 0
    };
  })()`,
  awaitPromise: true,
  returnByValue: true,
});
const genericDocumentState = await command("Runtime.evaluate", {
  expression: `(() => {
    const current = parseDocumentMemory("【文档《任意报告.docx》·正文】\\n通用正文");
    const legacy = parseDocumentMemory("【文件《任意笔记.md》】\\n旧格式正文");
    return {
      current: current && {
        filename: current.name,
        section: current.section,
        extension: current.extension,
        content: current.content
      },
      legacy: legacy && {
        filename: legacy.name,
        section: legacy.section,
        extension: legacy.extension,
        content: legacy.content
      }
    };
  })()`,
  returnByValue: true,
});
socket.close();

console.log(JSON.stringify({
  upload: state.result.value,
  memory: memoryState.result.value,
  markdown: markdownState.result.value,
  genericDocuments: genericDocumentState.result.value,
}));
if (
  state.result.value.count !== 2
  || state.result.value.renderedItems !== 2
  || !state.result.value.draftVisible
  || state.result.value.sentFileMark !== "MD"
  || !state.result.value.hasStandaloneText
  || state.result.value.sentPrompt !== "请分析这些附件。"
  || !memoryState.result.value.found
  || !memoryState.result.value.initiallyCollapsed
  || !memoryState.result.value.expanded
  || !memoryState.result.value.contentVisible
  || !memoryState.result.value.dayCard
  || !memoryState.result.value.dayCollapsed
  || !markdownState.result.value.found
  || markdownState.result.value.filename !== "讲稿.md"
  || markdownState.result.value.section !== "全文"
  || !markdownState.result.value.collapsed
  || markdownState.result.value.contentLength < 500
  || genericDocumentState.result.value.current?.filename !== "任意报告.docx"
  || genericDocumentState.result.value.current?.section !== "正文"
  || genericDocumentState.result.value.current?.extension !== "DOCX"
  || genericDocumentState.result.value.current?.content !== "通用正文"
  || genericDocumentState.result.value.legacy?.filename !== "任意笔记.md"
  || genericDocumentState.result.value.legacy?.section !== "全文"
  || genericDocumentState.result.value.legacy?.extension !== "MD"
  || genericDocumentState.result.value.legacy?.content !== "旧格式正文"
) {
  process.exitCode = 1;
}
