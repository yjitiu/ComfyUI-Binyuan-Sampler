// Binyuan 采样器 LoRA 管理界面（v5.5 DOM 覆盖层范式）
// 关键：整个 LoRA 管理器是一个 addDOMWidget，serialize:false —— 不进入 widgets_values，
// 不会污染 宽度/高度/生成数量 等原生 widget 的按位序列化（这是之前错乱/跑不起来的根因）。
// LoRA 数据同步到隐藏的原生输入口 lora_json（会被序列化并发往后端），并用 properties 持久化。
// 绝不使用 addWidget 动态加按钮/下拉/数值。
import { app } from "/scripts/app.js";

const NODE_NAME = "BinyuanUltimateSampler";
const WIDGET_NAME = "binyuan_lora_mgr";
const STYLE_ID = "binyuan-lora-mgr-style";

// 注入样式（只一次）
function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.bq-lora-wrap{display:flex;flex-direction:column;gap:4px;padding:6px 8px;font-size:11px;color:#ddd;}
.bq-lora-row{display:flex;align-items:center;gap:4px;}
.bq-lora-row select{flex:1 1 auto;min-width:0;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:3px;padding:2px 4px;font-size:11px;max-width:210px;}
.bq-lora-row input[type=number]{width:52px;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:3px;padding:2px 3px;font-size:11px;text-align:center;}
.bq-lora-row input[type=checkbox]{margin:0;flex:0 0 auto;}
.bq-lora-row button,.bq-lora-add{cursor:pointer;border:1px solid #555;border-radius:3px;background:#3a3a3a;color:#eee;padding:2px 6px;font-size:11px;}
.bq-lora-row button:hover,.bq-lora-add:hover{background:#4a6a9a;}
.bq-lora-row.bq-disabled{opacity:0.45;}
.bq-lora-add{align-self:flex-start;margin-top:2px;}
`;
    document.head.appendChild(st);
}

// 读取 lora 文件名列表
function getLoraOptionsSync(node) {
    try {
        const w = node.widgets?.find(x => x.name === "LORA_LIST");
        if (w?.options?.values?.length > 1) return w.options.values.slice();
    } catch (e) {}
    return ["None"];
}

async function fetchLoraOptionsFallback() {
    for (const url of ["/api/object_info", "/object_info"]) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) continue;
            const data = await resp.json();
            const arr = data?.[NODE_NAME]?.input?.required?.LORA_LIST;
            if (Array.isArray(arr) && Array.isArray(arr[0]) && arr[0].length > 1) return arr[0].slice();
        } catch (e) {}
    }
    return ["None"];
}

function hideNativeWidget(node, name) {
    try {
        const w = node.widgets?.find(x => x.name === name);
        if (w) { w.hidden = true; } // 仅隐藏，保留序列化与发送
    } catch (e) {}
}

function getJsonWidget(node) {
    return node.widgets?.find(w => w.name === "lora_json");
}

function readLoraJson(node) {
    try {
        const v = getJsonWidget(node)?.value;
        if (v && v !== "[]") {
            const arr = JSON.parse(v);
            if (Array.isArray(arr)) return arr;
        }
    } catch (e) {}
    // 回退到 properties
    try {
        const p = node.properties?.bq_loras;
        if (Array.isArray(p) && p.length) return p;
    } catch (e) {}
    return [];
}

function writeLoraJson(node, loras) {
    const filtered = loras.filter(l => l && l.n && l.n !== "None");
    const json = JSON.stringify(filtered.map(l => ({ n: l.n, s: l.s, e: l.e !== false })));
    const w = getJsonWidget(node);
    if (w) w.value = json;
    try { node.properties = node.properties || {}; node.properties.bq_loras = filtered; } catch (e) {}
    try { node.setDirtyCanvas(true, true); } catch (e) {}
}

function createRow(node, mgr, loraOptions, data) {
    const row = document.createElement("div");
    row.className = "bq-lora-row";

    const en = document.createElement("input");
    en.type = "checkbox";
    en.checked = data.e !== false;
    en.title = "启用/禁用";

    const sel = document.createElement("select");
    for (const opt of loraOptions) {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        sel.appendChild(o);
    }
    sel.value = (data.n && loraOptions.includes(data.n)) ? data.n : "None";

    const str = document.createElement("input");
    str.type = "number";
    str.step = "0.05"; str.min = "-2"; str.max = "2";
    str.value = (data.s == null ? 1.0 : data.s);

    const del = document.createElement("button");
    del.textContent = "🗑";
    del.title = "删除";

    row.append(en, sel, str, del);

    const sync = () => {
        if (en.checked) row.classList.remove("bq-disabled");
        else row.classList.add("bq-disabled");
        writeLoraJson(node, collectRows(mgr));
        resize(node, mgr);
    };

    en.addEventListener("change", sync);
    sel.addEventListener("change", sync);
    str.addEventListener("input", sync);
    del.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        row.remove();
        writeLoraJson(node, collectRows(mgr));
        resize(node, mgr);
    });
    // 阻止交互时拖动画布
    for (const evName of ["pointerdown", "mousedown", "wheel"]) {
        row.addEventListener(evName, (e) => e.stopPropagation());
    }

    if (!en.checked) row.classList.add("bq-disabled");
    return row;
}

function collectRows(mgr) {
    const out = [];
    mgr.wrap.querySelectorAll(".bq-lora-row").forEach(row => {
        const en = row.querySelector('input[type=checkbox]');
        const sel = row.querySelector('select');
        const str = row.querySelector('input[type=number]');
        out.push({ n: sel?.value || "None", s: parseFloat(str?.value) || 0, e: !!(en?.checked) });
    });
    return out;
}

function rebuildRows(node, mgr, loraOptions) {
    // 清空旧行（保留 add 按钮）
    mgr.wrap.querySelectorAll(".bq-lora-row").forEach(r => r.remove());
    const saved = readLoraJson(node);
    if (saved.length) {
        saved.forEach(s => mgr.wrap.insertBefore(createRow(node, mgr, loraOptions, s), mgr.addBtn));
    } else {
        mgr.wrap.insertBefore(createRow(node, mgr, loraOptions, { n: "None", s: 1.0, e: true }), mgr.addBtn);
    }
    resize(node, mgr);
}

function resize(node, mgr) {
    const rows = mgr.wrap.querySelectorAll(".bq-lora-row").length;
    const h = rows * 26 + 40;
    try {
        if (mgr.widget) {
            mgr.widget.getHeight = () => h;
            mgr.widget.computedHeight = h;
            mgr.widget.computeSize = (w) => [w || node.size?.[0] || 300, h];
        }
        node.setDirtyCanvas(true, true);
    } catch (e) {}
}

function ensureManager(node) {
    if (node._bqLoraMgr) return node._bqLoraMgr;
    ensureStyle();
    hideNativeWidget(node, "lora_json");
    hideNativeWidget(node, "LORA_LIST");

    const wrap = document.createElement("div");
    wrap.className = "bq-lora-wrap";
    for (const evName of ["pointerdown", "mousedown", "wheel", "contextmenu"]) {
        wrap.addEventListener(evName, (e) => e.stopPropagation());
    }

    const addBtn = document.createElement("button");
    addBtn.className = "bq-lora-add";
    addBtn.textContent = "+ 添加 LoRA";
    addBtn.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const mgr = node._bqLoraMgr;
        wrap.insertBefore(createRow(node, mgr, mgr.loraOptions, { n: "None", s: 1.0, e: true }), addBtn);
        writeLoraJson(node, collectRows(mgr));
        resize(node, mgr);
    });
    wrap.appendChild(addBtn);

    const widget = node.addDOMWidget?.(WIDGET_NAME, "HTML", wrap, {
        hideOnZoom: false,
        serialize: false,
        getHeight: () => 40,
    });
    if (widget) {
        widget.serialize = false;
        widget.computeSize = (w) => [w || node.size?.[0] || 300, 40];
    }

    const mgr = { wrap, addBtn, widget, loraOptions: getLoraOptionsSync(node) };
    node._bqLoraMgr = mgr;

    // 兜底：同步取不到列表时异步拉取并刷新下拉
    if (mgr.loraOptions.length <= 1) {
        fetchLoraOptionsFallback().then(opts => {
            if (opts && opts.length > 1) {
                mgr.loraOptions = opts;
                mgr.wrap.querySelectorAll(".bq-lora-row select").forEach(sel => {
                    const cur = sel.value;
                    sel.innerHTML = "";
                    for (const opt of opts) {
                        const o = document.createElement("option");
                        o.value = opt; o.textContent = opt;
                        sel.appendChild(o);
                    }
                    sel.value = opts.includes(cur) ? cur : "None";
                });
            }
        }).catch(() => {});
    }
    return mgr;
}

try {
    app.registerExtension({
        name: "BinyuanUltimateSampler.LoraUI",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData?.name !== NODE_NAME) return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const node = this;
                try {
                    const mgr = ensureManager(node);
                    rebuildRows(node, mgr, mgr.loraOptions);
                } catch (e) {
                    console.warn("[Binyuan] LoRA 管理器初始化失败：", e);
                }
            };

            // 工作流加载后，lora_json 的值才被恢复，这里据此重建行
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                const node = this;
                try {
                    const mgr = node._bqLoraMgr || ensureManager(node);
                    rebuildRows(node, mgr, mgr.loraOptions);
                } catch (e) {}
                return r;
            };

            // 尺寸助手联动（原生 combo，不污染 widgets_values）
            const onAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function () {
                onAdded?.apply(this, arguments);
                const node = this;
                setTimeout(() => {
                    try {
                        const sizeHelper = node.widgets?.find(w => w.name === "尺寸助手");
                        const widthWidget = node.widgets?.find(w => w.name === "宽度");
                        const heightWidget = node.widgets?.find(w => w.name === "高度");
                        if (sizeHelper && widthWidget && heightWidget) {
                            sizeHelper.callback = (v) => {
                                const m = String(v).match(/(\d+)x(\d+)/);
                                if (m) { widthWidget.value = parseInt(m[1]); heightWidget.value = parseInt(m[2]); }
                                node.setDirtyCanvas(true, true);
                            };
                        }
                        const posWidget = node.widgets?.find(w => w.name === "正面提示词");
                        const negWidget = node.widgets?.find(w => w.name === "负面提示词");
                        if (posWidget?.inputEl) { posWidget.inputEl.style.background = "#1a331a"; posWidget.inputEl.style.color = "#fff"; }
                        if (negWidget?.inputEl) { negWidget.inputEl.style.background = "#331a1a"; negWidget.inputEl.style.color = "#fff"; }
                    } catch (e) {}
                }, 100);
            };
        },
    });
} catch (e) {
    console.warn("[Binyuan] LoRA 扩展注册失败：", e);
}
