import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "BinyuanUltimateSampler.JS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "BinyuanUltimateSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                const self = this;
                const getWidget = (name) => self.widgets?.find(w => w.name === name);
                
                // ========== 获取 LoRA 列表选项 ==========
                const loraListWidget = getWidget("LORA_LIST");
                let loraOptions = ["None"];
                if (loraListWidget && loraListWidget.options && loraListWidget.options.values) {
                    loraOptions = loraListWidget.options.values;
                }
                
                // ========== 隐藏原始 LoRA 相关 widget ==========
                const hideWidgets = ["lora_json", "LORA_LIST"];
                for (const name of hideWidgets) {
                    const w = getWidget(name);
                    if (w) {
                        w.hidden = true;
                        w.type = "hidden";
                    }
                }
                
                // ========== LoRA 多行管理 ==========
                if (!self.loraRows) self.loraRows = [];
                
                const updateLoraJson = () => {
                    const loras = self.loraRows
                        .filter(row => row.enabled && row.name && row.name !== "None")
                        .map(row => ({ n: row.name, s: row.strength, e: true }));
                    const jsonWidget = getWidget("lora_json");
                    if (jsonWidget) {
                        jsonWidget.value = JSON.stringify(loras);
                    }
                };
                
                // 删除所有动态添加的 LoRA widget
                const dynamicWidgets = self.widgets.filter(w => 
                    w.name === "lora_enable_btn" || 
                    w.name === "lora_name_combo" || 
                    w.name === "lora_strength" || 
                    w.name === "lora_delete_btn" ||
                    w.name === "lora_add_btn"
                );
                dynamicWidgets.forEach(w => {
                    const idx = self.widgets.indexOf(w);
                    if (idx !== -1) self.widgets.splice(idx, 1);
                });
                self.loraRows = [];
                
                const createLoraRow = (defaultName = "None", defaultStrength = 1.0, defaultEnabled = true) => {
                    const rowId = Date.now() + Math.random();
                    const rowWidgets = [];
                    
                    const enableText = defaultEnabled ? "✓启用" : "✗禁用";
                    const enableBtn = self.addWidget("button", enableText, null, () => {
                        const row = self.loraRows.find(r => r.rowId === rowId);
                        if (row) {
                            row.enabled = !row.enabled;
                            const newText = row.enabled ? "✓启用" : "✗禁用";
                            enableBtn.name = newText;
                            if (enableBtn.element) {
                                enableBtn.element.innerText = newText;
                            }
                            updateLoraJson();
                        }
                        self.setDirtyCanvas(true, true);
                    });
                    enableBtn.options = { width: 56 };
                    rowWidgets.push(enableBtn);
                    
                    const nameCombo = self.addWidget("combo", "", defaultName, (v) => {
                        const row = self.loraRows.find(r => r.rowId === rowId);
                        if (row) {
                            row.name = v;
                            updateLoraJson();
                        }
                    }, { values: loraOptions });
                    nameCombo.options.width = 180;
                    rowWidgets.push(nameCombo);
                    
                    const strengthWidget = self.addWidget("number", "", defaultStrength, (v) => {
                        const row = self.loraRows.find(r => r.rowId === rowId);
                        if (row) {
                            row.strength = v;
                            updateLoraJson();
                        }
                    }, { min: -2, max: 2, step: 0.05 });
                    strengthWidget.options.width = 70;
                    rowWidgets.push(strengthWidget);
                    
                    const delBtn = self.addWidget("button", "🗑删除", null, () => {
                        rowWidgets.forEach(w => {
                            const idx = self.widgets.indexOf(w);
                            if (idx !== -1) self.widgets.splice(idx, 1);
                        });
                        const rowIdx = self.loraRows.findIndex(r => r.rowId === rowId);
                        if (rowIdx !== -1) self.loraRows.splice(rowIdx, 1);
                        updateLoraJson();
                        self.setSize([self.size[0], self.computeSize()[1]]);
                        self.setDirtyCanvas(true, true);
                    });
                    delBtn.options = { width: 56 };
                    rowWidgets.push(delBtn);
                    
                    self.loraRows.push({
                        rowId: rowId,
                        name: defaultName,
                        strength: defaultStrength,
                        enabled: defaultEnabled,
                        widgets: rowWidgets
                    });
                    
                    return rowWidgets;
                };
                
                let addBtn = self.widgets.find(w => w.name === "lora_add_btn");
                if (!addBtn) {
                    addBtn = self.addWidget("button", "+ 添加 LoRA", null, () => {
                        createLoraRow();
                        updateLoraJson();
                        self.setSize([self.size[0], self.computeSize()[1] + 45]);
                        self.setDirtyCanvas(true, true);
                    });
                    addBtn.name = "lora_add_btn";
                }
                
                const savedJsonWidget = getWidget("lora_json");
                if (savedJsonWidget && savedJsonWidget.value && savedJsonWidget.value !== "[]") {
                    try {
                        const savedList = JSON.parse(savedJsonWidget.value);
                        if (Array.isArray(savedList) && savedList.length > 0) {
                            savedList.forEach(lora => {
                                if (lora.n && lora.n !== "None") {
                                    createLoraRow(lora.n, lora.s || 1.0, lora.e !== false);
                                }
                            });
                        }
                    } catch(e) {}
                }
                
                if (self.loraRows.length === 0) {
                    createLoraRow();
                }
                
                // ========== 重新排序 widgets ==========
                const widgetOrder = [
                    "加载模式",
                    "Checkpoint",
                    "扩散模型",
                    "CLIP_1",
                    "CLIP_2",
                    "CLIP_类型",
                    "VAE",
                    "权重精度",
                    "串联模式",
                    "上游图像处理",
                    "Latent输入源",
                    "正面提示词",
                    "负面提示词",
                    "尺寸助手",
                    "宽度",
                    "高度",
                    "生成数量",
                    "seed",
                    "步数",
                    "CFG",
                    "Flux引导",
                    "采样算法",
                    "调度器",
                    "重绘强度"
                ];
                
                for (let i = 0; i < widgetOrder.length; i++) {
                    const w = getWidget(widgetOrder[i]);
                    if (w) {
                        const currentIdx = self.widgets.indexOf(w);
                        if (currentIdx !== -1 && currentIdx !== i) {
                            self.widgets.splice(currentIdx, 1);
                            self.widgets.splice(i, 0, w);
                        }
                    }
                }
                
                // ========== 交换宽高按钮 ==========
                const sizeHelper = getWidget("尺寸助手");
                const widthWidget = getWidget("宽度");
                const heightWidget = getWidget("高度");
                
                if (sizeHelper && widthWidget && heightWidget) {
                    sizeHelper.callback = (v) => {
                        const m = v.match(/(\d+)x(\d+)/);
                        if (m) {
                            widthWidget.value = parseInt(m[1]);
                            heightWidget.value = parseInt(m[2]);
                        }
                        self.setDirtyCanvas(true, true);
                    };
                    
                    let swapBtn = self.widgets?.find(w => w.name === "🔄 交换宽高");
                    if (!swapBtn) {
                        swapBtn = self.addWidget("button", "🔄 交换宽高", "", () => {
                            const temp = widthWidget.value;
                            widthWidget.value = heightWidget.value;
                            heightWidget.value = temp;
                            self.setDirtyCanvas(true, true);
                        });
                    }
                    
                    const sizeHelperIdx = self.widgets.indexOf(sizeHelper);
                    const swapIdx = self.widgets.indexOf(swapBtn);
                    
                    if (swapIdx !== -1) {
                        self.widgets.splice(swapIdx, 1);
                    }
                    if (sizeHelperIdx !== -1) {
                        self.widgets.splice(sizeHelperIdx + 1, 0, swapBtn);
                    }
                }
                
                // 提示词样式
                const posWidget = getWidget("正面提示词");
                const negWidget = getWidget("负面提示词");
                if (posWidget && posWidget.inputEl) {
                    posWidget.inputEl.style.background = "#1a331a";
                    posWidget.inputEl.style.color = "#ffffff";
                }
                if (negWidget && negWidget.inputEl) {
                    negWidget.inputEl.style.background = "#331a1a";
                    negWidget.inputEl.style.color = "#ffffff";
                }
                
                setTimeout(() => {
                    self.setSize([self.size[0], self.computeSize()[1]]);
                    self.setDirtyCanvas(true, true);
                }, 200);
            };
        }
    }
});
