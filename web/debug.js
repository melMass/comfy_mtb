import { app } from "/scripts/app.js";
import * as shared from '/extensions/mtb/comfy_shared.js'
import { log } from '/extensions/mtb/comfy_shared.js'

// TODO: respect inputs order...


const DEBUG_WIDGETS = {
    "image": (val, index) => {
        const w = {
            name: `anything_${index}`,
            type: "image",
            value: val,
            draw: function (ctx,
                node,
                widgetWidth,
                widgetY,
                height) {
                const [cw, ch] = this.computeSize(widgetWidth)
                shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
            },
            computeSize: function (width) {
                const ratio = this.inputRatio || 1;
                if (width) {
                    return [width, width / ratio + 4]
                }
                return [128, 128]
            },
            onRemove: function () {
                if (this.inputEl) {
                    this.inputEl.remove();
                }
            }
        }

        w.inputEl = document.createElement("img");
        w.inputEl.src = "data:image/jpeg;base64," + w.value;
        w.inputEl.onload = function () {
            w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight;
        }
        document.body.appendChild(w.inputEl);
        return w
    },
    "text": (val, index) => {
        const w = {
            name: `anything_${index}`,
            type: "debug_text",
            val: val,
            draw: function (ctx,
                node,
                widgetWidth,
                widgetY,
                height) {
                // const [cw, ch] = this.computeSize(widgetWidth)
                shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, height)
            },
            computeSize: function (width) {
                const value = this.inputEl.innerHTML
                if (!value) {
                    return [32, 32]
                }
                if (!width) {
                    log(`No width ${this.parent.size}`)
                }

                const fontSize = 25; // Assuming 1rem = 16px

                const oldFont = app.ctx.font
                app.ctx.font = `${fontSize}px Arial`;

                const words = value.split(" ");
                const lines = [];
                let currentLine = "";
                for (const word of words) {
                    const testLine = currentLine.length === 0 ? word : `${currentLine} ${word}`;

                    const testWidth = app.ctx.measureText(testLine).width;

                    // log(`Testing line ${testLine}, width: ${testWidth}, width: ${width}, ratio: ${testWidth / width}`)
                    if (testWidth > width) {
                        lines.push(currentLine);
                        currentLine = word;
                    } else {
                        currentLine = testLine;
                    }
                }
                app.ctx.font = oldFont;
                lines.push(currentLine);

                // Step 3: Calculate the widget width and height
                const textHeight = lines.length * (fontSize + 2); // You can adjust the line height (2 in this case)
                const maxLineWidth = lines.reduce((maxWidth, line) => Math.max(maxWidth, app.ctx.measureText(line).width), 0);
                const widgetWidth = Math.max(width || this.width || 32, maxLineWidth);
                const widgetHeight = textHeight + 10; // Additional padding for spacing
                return [widgetWidth, widgetHeight + 4]

            },
            onRemove: function () {
                if (this.inputEl) {
                    this.inputEl.remove();
                }

            }
        }
        w.inputEl = document.createElement("p");
        w.inputEl.style.textAlign = "center";
        w.inputEl.style.fontSize = "1.5em";
        w.inputEl.style.color = "var(--input-text)";
        w.inputEl.style.fontFamily = "monospace";
        w.inputEl.innerHTML = val
        document.body.appendChild(w.inputEl);

        return w
    }
}

app.registerExtension({
    name: "mtb.Debug",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Debug (mtb)") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                // TODO: remove all widgets on disconnect once computed
                shared.dynamic_connection(this, index, connected, "anything_", "*")

                //- infer type
                if (link_info) {
                    const fromNode = this.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
                    const type = fromNode.outputs[link_info.origin_slot].type;
                    this.inputs[index].type = type;
                    // this.inputs[index].label = type.toLowerCase()
                }
                //- restore dynamic input
                if (!connected) {
                    this.inputs[index].type = "*";
                    this.inputs[index].label = `anything_${index + 1}`
                }
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                log(message)
                onExecuted?.apply(this, arguments);
                log(message)
                if (this.widgets) {
                    // const pos = this.widgets.findIndex((w) => w.name === "anything_1");
                    // if (pos !== -1) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = 0;

                }
                let widgetI = 1
                if (message.text) {
                    for (const txt of message.text) {
                        const w = this.addCustomWidget(DEBUG_WIDGETS["text"](txt, widgetI))
                        w.parent = this;
                        widgetI++;
                    }
                }
                if (message.b64_images) {
                    for (const img of message.b64_images) {
                        const w = this.addCustomWidget(DEBUG_WIDGETS["image"](img, widgetI))
                        w.parent = this;
                        widgetI++;
                    }
                    // this.onResize?.(this.size);
                    // this.resize?.(this.size)
                    this.setSize(this.computeSize())
                };

                this.onRemoved = function () {
                    // When removing this node we need to remove the input from the DOM
                    for (let y in this.widgets) {
                        if (this.widgets[y].canvas) {
                            this.widgets[y].canvas.remove();
                        }
                        this.widgets[y].onRemove?.();
                    }
                }
            }
        }
    }
}
);
