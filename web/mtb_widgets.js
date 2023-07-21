import { app } from "/scripts/app.js";
import parseCss from '/extensions/mtb/extern/parse-css.js'
import * as shared from '/extensions/mtb/comfy_shared.js'

import { api } from "/scripts/api.js";

import { ComfyWidgets } from "/scripts/widgets.js";

const newTypes = ["BOOL", "COLOR", "BBOX"]

const bboxWidget = (key, val) => {
    /** @type {import("./types/litegraph").IWidget} */
    const widget = {
        name: key,
        type: "BBOX",
        // options: val,
        y: 0,
        value: val || [0, 0, 0, 0],
        options: {},

        draw: function (ctx,
            node,
            widget_width,
            widgetY,
            height) {
            const hide = this.type !== "BBOX" && app.canvas.ds.scale > 0.5

            const show_text = true
            const outline_color = LiteGraph.WIDGET_OUTLINE_COLOR;
            const background_color = LiteGraph.WIDGET_BGCOLOR;
            const text_color = LiteGraph.WIDGET_TEXT_COLOR;
            const secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
            const H = LiteGraph.NODE_WIDGET_HEIGHT;


            var margin = 15;
            if (hide) return;

            ctx.textAlign = "left";
            ctx.strokeStyle = outline_color;
            ctx.fillStyle = background_color;
            ctx.beginPath();
            if (show_text)
                ctx.roundRect(margin, widgetY, widget_width - margin * 2, H, [H * 0.5]);
            else
                ctx.rect(margin, widgetY, widget_width - margin * 2, H);
            ctx.fill();
            if (show_text) {
                if (!this.disabled)
                    ctx.stroke();
                ctx.fillStyle = text_color;
                if (!this.disabled) {
                    ctx.beginPath();
                    ctx.moveTo(margin + 16, widgetY + 5);
                    ctx.lineTo(margin + 6, widgetY + H * 0.5);
                    ctx.lineTo(margin + 16, widgetY + H - 5);
                    ctx.fill();
                    ctx.beginPath();
                    ctx.moveTo(widget_width - margin - 16, widgetY + 5);
                    ctx.lineTo(widget_width - margin - 6, widgetY + H * 0.5);
                    ctx.lineTo(widget_width - margin - 16, widgetY + H - 5);
                    ctx.fill();
                }
                ctx.fillStyle = secondary_text_color;
                ctx.fillText(this.label || this.name, margin * 2 + 5, widgetY + H * 0.7);
                ctx.fillStyle = text_color;
                ctx.textAlign = "right";

                ctx.fillText(
                    Number(this.value).toFixed(
                        this.options?.precision !== undefined
                            ? this.options.precision
                            : 3
                    ),
                    widget_width - margin * 2 - 20,
                    widgetY + H * 0.7
                );

            }
        },
        mouse: function (event, pos, node) {
            var old_value = this.value;
            var x = pos[0] - node.pos[0];
            var y = pos[1] - node.pos[1];
            var width = node.size[0];

            if (event.type == LiteGraph.pointerevents_method + "move" && this.type == "BBOX") {
                if (event.deltaX)
                    this.value += event.deltaX * 0.1 * (this.options?.step || 1);
                if (this.options.min != null && this.value < this.options.min) {
                    this.value = this.options.min;
                }
                if (this.options.max != null && this.value > this.options.max) {
                    this.value = this.options.max;
                }
            } else if (event.type == LiteGraph.pointerevents_method + "down") {
                var values = this.options?.values;
                if (values && values.constructor === Function) {
                    values = this.options.values(w, node);
                }
                var values_list = null;

                var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                if (this.type == "BBOX") {
                    this.value += delta * 0.1 * (this.options.step || 1);
                    if (this.options.min != null && this.value < this.options.min) {
                        this.value = this.options.min;
                    }
                    if (this.options.max != null && this.value > this.options.max) {
                        this.value = this.options.max;
                    }
                } else if (delta) { //clicked in arrow, used for combos 
                    var index = -1;
                    this.last_mouseclick = 0; //avoids dobl click event
                    if (values.constructor === Object)
                        index = values_list.indexOf(String(this.value)) + delta;
                    else
                        index = values_list.indexOf(this.value) + delta;
                    if (index >= values_list.length) {
                        index = values_list.length - 1;
                    }
                    if (index < 0) {
                        index = 0;
                    }
                    if (values.constructor === Array)
                        this.value = values[index];
                    else
                        this.value = index;
                } else { //combo clicked 
                    var text_values = values != values_list ? Object.values(values) : values;
                    var menu = new LiteGraph.ContextMenu(text_values, {
                        scale: Math.max(1, this.ds.scale),
                        event: event,
                        className: "dark",
                        callback: inner_clicked.bind(w)
                    },
                        ref_window);
                    function inner_clicked(v, option, event) {
                        if (values != values_list)
                            v = text_values.indexOf(v);
                        this.value = v;
                        shared.inner_value_change(this, v, event);
                        app.canvas.setDirty(true)
                        return false;
                    }
                }
            } //end mousedown
            else if (event.type == LiteGraph.pointerevents_method + "up" && this.type == "BBOX") {
                var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                if (event.click_time < 200 && delta == 0) {
                    this.prompt("Value", this.value, function (v) {
                        // check if v is a valid equation or a number
                        if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                            try {//solve the equation if possible
                                v = eval(v);
                            } catch (e) { }
                        }
                        this.value = Number(v);
                        shared.inner_value_change(this, this.value, event);
                    }.bind(w),
                        event);
                }
            }

            if (old_value != this.value)
                setTimeout(
                    function () {
                        shared.inner_value_change(this, this.value, event);
                    }.bind(this),
                    20
                );

            app.canvas.setDirty(true)

        },
        computeSize: function (width) {
            return [width, 32];
        },
        // onDrawBackground: function (ctx) {
        //     if (!this.flags.collapsed) return;
        //     this.inputEl.style.display = "block";
        //     this.inputEl.style.top = this.graphcanvas.offsetTop + this.pos[1] + "px";
        //     this.inputEl.style.left = this.graphcanvas.offsetLeft + this.pos[0] + "px";
        // },
        // onInputChange: function (e) {
        //     const property = e.target.dataset.property;
        //     const bbox = this.getInputData(0);
        //     if (!bbox) return;
        //     bbox[property] = parseFloat(e.target.value);
        //     this.setOutputData(0, bbox);
        // }
    }

    widget.desc = "Represents a Bounding Box with x, y, width, and height.";
    return widget

}
const bboxWidgetDOM = (key, val) => {
    /** @type {import("./types/litegraph").IWidget} */
    const widget = {
        name: key,
        type: "BBOX",
        // options: val,
        y: 0,
        value: val || [0, 0, 0, 0],

        draw: function (ctx,
            node,
            widgetWidth,
            widgetY,
            height) {
            const hide = this.type !== "BBOX" && app.canvas.ds.scale > 0.5
            this.inputEl.style.display = hide ? "none" : "block";
            if (hide) return;

            shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, height)
        },
        computeSize: function (width) {
            return [width, 32];
        },
        // onDrawBackground: function (ctx) {
        //     if (!this.flags.collapsed) return;
        //     this.inputEl.style.display = "block";
        //     this.inputEl.style.top = this.graphcanvas.offsetTop + this.pos[1] + "px";
        //     this.inputEl.style.left = this.graphcanvas.offsetLeft + this.pos[0] + "px";
        // },
        onInputChange: function (e) {
            const property = e.target.dataset.property;
            const bbox = this.getInputData(0);
            if (!bbox) return;
            bbox[property] = parseFloat(e.target.value);
            this.setOutputData(0, bbox);
        }
    }
    widget.inputEl = document.createElement("div")
    widget.parent = this
    widget.inputEl.innerHTML = `
    <div><input type="number" step="1" class="bbox-input" data-property="x" placeholder="x" /></div>
    <div><input type="number" step="1" class="bbox-input" data-property="y" placeholder="y" /></div>
    <div><input type="number" step="1" class="bbox-input" data-property="width" placeholder="width" /></div>
    <div><input type="number" step="1" class="bbox-input" data-property="height" placeholder="height" /></div>
  `;
    // set the class document wide if not present

    shared.defineClass("bbox-input", `background-color: var(--comfy-input-bg);
	color: var(--input-text);
	overflow: hidden;
    width:100%;
	overflow-y: auto;
	padding: 2px;
	resize: none;
	border: none;
	box-sizing: border-box;
	font-size: 10px;`)


    const bboxInputs = widget.inputEl.querySelectorAll(".bbox-input");
    bboxInputs.forEach((input) => {
        input.addEventListener("change", widget.onInputChange.bind(this));
    });

    widget.desc = "Represents a Bounding Box with x, y, width, and height.";

    document.body.appendChild(widget.inputEl);


    console.log(widget.inputEl)
    return widget

}
/**
 * @returns {import("./types/litegraph").IWidget} widget
 */
const boolWidget = (key, val, compute = false) => {
    /** @type {import("/types/litegraph").IWidget} */
    const widget = {
        name: key,
        type: "BOOL",
        options: { default: false },
        y: 0,
        value: val || false,
        draw: function (ctx,
            node,
            widgetWidth,
            widgetY,
            height) {
            const hide = this.type !== "BOOL"
            if (hide) {
                return
            }
            const border = 6;
            const w = 20;
            ctx.fillStyle = "black";

            ctx.fillRect(24, widgetY, w, w);
            if (this.value) {
                ctx.fillStyle = "white";
                ctx.fillRect(24 + border, widgetY + border, w - border * 2, height - border * 2);
            }
            console.log(typeof (this.value))
            // write the input name
            // choose the fill based on the luminoisty of this.value color

            ctx.fillStyle = "white";


            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.name, widgetWidth * 0.5, widgetY + 14);

        },
        get value() {

            return this.inputEl.value === "true";
        },
        set value(x) {
            this.inputEl.value = x;
        },
        computeSize: function (width) {
            return [width, 32];
        },
        mouse: function (e, pos, node) {
            if (e.type === "pointerdown") {
                // get widgets of type type : "COLOR"
                const widgets = node.widgets.filter(w => w.type === "BOOL");

                for (const w of widgets) {
                    // color picker
                    const rect = [w.last_y, w.last_y + 32];
                    if (pos[1] > rect[0] && pos[1] < rect[1]) {
                        console.log("bool", this.value)
                        // picker.style.position = "absolute";
                        // picker.style.left = ( pos[0]) + "px";
                        // picker.style.top = (  pos[1]) + "px";

                        // place at screen center
                        // picker.style.position = "absolute";
                        // picker.style.left = (window.innerWidth / 2) + "px";
                        // picker.style.top = (window.innerHeight / 2) + "px";
                        // picker.style.transform = "translate(-50%, -50%)";
                        // picker.style.zIndex = 1000;

                        this.value = this.value ? "false" : "true"

                    }
                }
            }
        }
    }

    // create a checkbox
    widget.inputEl = document.createElement("input")
    widget.inputEl.type = "checkbox"
    widget.inputEl.value = false
    document.body.appendChild(widget.inputEl);
    return widget

}

/**
 * @returns {import("./types/litegraph").IWidget} widget
 */
const colorWidget = (key, val, compute = false) => {
    /** @type {import("/types/litegraph").IWidget} */
    const widget = {}
    widget.y = 0
    widget.name = key;
    widget.type = "COLOR";
    widget.options = { default: "#ff0000" };
    widget.value = val || "#ff0000";
    widget.draw = function (ctx,
        node,
        widgetWidth,
        widgetY,
        height) {
        const hide = this.type !== "COLOR" && app.canvas.ds.scale > 0.5
        if (hide) {
            return
        }

        const border = 3;
        // draw a rect with a border and a fill color
        ctx.fillStyle = "#000";
        ctx.fillRect(0, widgetY, widgetWidth, height);
        ctx.fillStyle = this.value;
        ctx.fillRect(border, widgetY + border, widgetWidth - border * 2, height - border * 2);
        // write the input name
        // choose the fill based on the luminoisty of this.value color
        const color = parseCss(this.value.default || this.value)
        if (!color) {
            return
        }
        ctx.fillStyle = shared.isColorBright(color.values, 125) ? "#000" : "#fff";


        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.fillText(this.name, widgetWidth * 0.5, widgetY + 14);



        // ctx.strokeStyle = "#fff";
        // ctx.strokeRect(border, widgetY + border, widgetWidth - border * 2, height - border * 2);


        // ctx.fillStyle = "#000";
        // ctx.fillRect(widgetWidth/2 - border / 2 , widgetY + border / 2 , widgetWidth/2 + border / 2, height + border / 2);
        // ctx.fillStyle = this.value;
        // ctx.fillRect(widgetWidth/2, widgetY, widgetWidth/2, height);

    }
    widget.mouse = function (e, pos, node) {
        if (e.type === "pointerdown") {
            // get widgets of type type : "COLOR"
            const widgets = node.widgets.filter(w => w.type === "COLOR");

            for (const w of widgets) {
                // color picker
                const rect = [w.last_y, w.last_y + 32];
                if (pos[1] > rect[0] && pos[1] < rect[1]) {
                    console.log("color picker", node)
                    const picker = document.createElement("input");
                    picker.type = "color";
                    picker.value = this.value;
                    // picker.style.position = "absolute";
                    // picker.style.left = ( pos[0]) + "px";
                    // picker.style.top = (  pos[1]) + "px";

                    // place at screen center
                    picker.style.position = "absolute";
                    picker.style.left = "999999px"//(window.innerWidth / 2) + "px";
                    picker.style.top = "999999px" //(window.innerHeight / 2) + "px";
                    // picker.style.transform = "translate(-50%, -50%)";
                    // picker.style.zIndex = 1000;



                    document.body.appendChild(picker);

                    picker.addEventListener("change", () => {
                        this.value = picker.value;
                        node.graph._version++;
                        node.setDirtyCanvas(true, true);
                        picker.remove();
                    });

                    picker.click()

                }
            }
        }
    }
    widget.computeSize = function (width) {
        return [width, 32];
    }

    return widget;
}


// VIDEO: (node, inputName, inputData, app) => {
//             console.log("video")
//             const videoWidget = {
//                 name: "VideoWidget",
//                 description: "Video Player Widget",
//                 value: inputData,
//                 properties: {},
//                 widget: null,

//                 init: function () {
//                     this.widget = document.createElement("video");
//                     this.widget.width = 200;
//                     this.widget.height = 120;
//                     this.widget.controls = true;
//                     this.widget.style.width = "100%";
//                     this.widget.style.height = "100%";
//                     this.widget.style.objectFit = "contain";
//                     this.widget.style.backgroundColor = "black";
//                     this.widget.style.pointerEvents = "none";
//                     node.addWidget(inputName, videoWidget.widget, inputData);
//                 },

//                 setValue: function (value, options) {
//                     if (value instanceof HTMLVideoElement) {
//                         this.widget.src = value.src;
//                     } else if (typeof value === "string") {
//                         this.widget.src = value;
//                     }
//                 },

//                 getValue: function () {
//                     return this.widget.src;
//                 },

//                 append: function (parent) {
//                     parent.appendChild(this.widget);
//                 },

//                 remove: function () {
//                     this.widget.parentNode.removeChild(this.widget);
//                 }
//             };
//             return {
//                 widget: videoWidget,
//             }
//         }



/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const mtb_widgets = {
    name: "mtb.widgets",

    init: () => { },


    /**
     * @param {import("./types/comfy").NodeType} nodeType
     * @param {import("./types/comfy").NodeDef} nodeData
     * @param {import("./types/comfy").App} app
     */
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        const rinputs = nodeData.input?.required;

        let has_custom = false
        if (nodeData.input && nodeData.input.required) {
            for (const i of Object.keys(nodeData.input.required)) {
                const input_type = nodeData.input.required[i][0]

                if (newTypes.includes(input_type)) {
                    has_custom = true
                    break;
                }
            }
        }
        if (has_custom) {

            //- Add widgets on node creation
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.serialize_widgets = true;
                for (const [key, input] of Object.entries(rinputs)) {
                    switch (input[0]) {
                        case "COLOR":
                            const colW = colorWidget(key, input[1])
                            this.addCustomWidget(colW)
                            const associated_input = this.inputs.findIndex((i) => i.widget?.name === key);
                            if (associated_input !== -1) {
                                this.inputs[associated_input].widget = colW
                            }



                            break;
                        case "BOOL":
                            // const widg = boolWidget(key, input[1])
                            // this.addCustomWidget(widg)
                            this.addWidget("toggle", key, false, function (value, widget, node) {
                                console.log(value)

                            })
                            //this.removeInput(this.inputs.findIndex((i) => i.widget?.name === key));

                            break;
                        case "BBOX":

                            // const x_widget = ComfyWidgets.FLOAT(this, `${key}_x`, [undefined, { default: 1, min: 1, max: Number.MAX_SAFE_INTEGER }]).widget;
                            // const y_widget = ComfyWidgets.FLOAT(this, `${key}_y`, [undefined, { default: 1, min: 1, max: Number.MAX_SAFE_INTEGER }]).widget;
                            // const width_widget = ComfyWidgets.FLOAT(this, `${key}_width`, [undefined, { default: 1, min: 1, max: Number.MAX_SAFE_INTEGER }]).widget;
                            // const height_widget = ComfyWidgets.FLOAT(this, `${key}_height`, [undefined, { default: 1, min: 1, max: Number.MAX_SAFE_INTEGER }]).widget;



                            // this.addWidget("number", `${key}_x`, false, function (value, widget, node) { })
                            // this.addWidget("number", `${key}_y`, false, function (value, widget, node) { })
                            // this.addWidget("number", `${key}_width`, false, function (value, widget, node) { })
                            // this.addWidget("number", `${key}_height`, false, function (value, widget, node) { })

                            const bboxW = bboxWidget(key, input[1])
                            this.addCustomWidget(bboxW)
                            break;
                        default:
                            break
                    }


                    // }
                }

                this.setSize?.(this.computeSize())

                this.onRemoved = function () {
                    // When removing this node we need to remove the input from the DOM
                    for (let y in this.widgets) {
                        if (this.widgets[y].canvas) {
                            this.widgets[y].canvas.remove();
                        }
                    }
                };
            }

            //- On initial configure of nodes hide all converted widgets
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                console.log(this)

                if (this.inputs) {
                    for (const input of this.inputs) {
                        console.log(input)
                        if (input.widget) {
                            // if (newTypes.includes(input.type)) {

                            console.log(input.widget)
                            if (input.widget.hidden) {
                                console.debug(`Already hidden skipping ${input.widget.name}`)
                                continue
                            }
                            const w = this.widgets.find((w) => w.name === input.widget.name);
                            if (w) {
                                console.log(`hidding ${w.name} from ${this.label}`)
                                shared.hideWidget(this, w);
                            } else {
                                console.log(`converting to widget ${w}`)

                                shared.convertToWidget(this, input)
                            }
                        }
                    }
                }

                return r;
            };


            //- Extra menus
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;
                if (this.widgets) {
                    let toInput = [];
                    let toWidget = [];
                    for (const w of this.widgets) {
                        if (w.type === shared.CONVERTED_TYPE) {
                            toWidget.push({
                                content: `Convert ${w.name} to widget`,
                                callback: () => shared.convertToWidget(this, w),
                            });
                        } else {
                            const config = nodeData?.input?.required[w.name] || nodeData?.input?.optional?.[w.name] || [w.type, w.options || {}];

                            toInput.push({
                                content: `Convert ${w.name} to input`,
                                callback: () => shared.convertToInput(this, w, config),
                            });
                        }
                    }
                    if (toInput.length) {
                        options.push(...toInput, null);
                    }

                    if (toWidget.length) {
                        options.push(...toWidget, null);
                    }
                }

                return r;
            };

        }



        // if (nodeData.input && nodeData.input.includes("BOOL")) {

        //     addMenuHandler(nodeType, function (_, options) {
        //         options.unshift({
        //             content: "convert bool input to widget",
        //             callback: () => {

        //                 console.log("Nahh not yet")
        //             },
        //         });
        //     });

        // }

        //- Extending Python Nodes
        switch (nodeData.name) {
            case "Psd Save (mtb)": {
                // const onConnectionsChange = nodeType.prototype.onConnectionsChange;
                nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                    // const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                    shared.dynamic_connection(this, index, connected)
                }
                break
            }
            case "Animation Builder (mtb)": {
                // console.log(nodeType.prototype)



                nodeType.prototype.onNodeCreated = function () {

                    this.changeMode(LiteGraph.ALWAYS)
                    // api.addEventListener("executed", ({ detail }) => {

                    //     console.log("executed", detail)
                    //     console.log(this)

                    // })
                    const raw_iteration = this.widgets.find((w) => w.name === "raw_iteration");
                    const raw_loop = this.widgets.find((w) => w.name === "raw_loop");


                    const total_frames = this.widgets.find((w) => w.name === "total_frames");
                    const loop_count = this.widgets.find((w) => w.name === "loop_count");

                    shared.hideWidgetForGood(this, raw_iteration);
                    shared.hideWidgetForGood(this, raw_loop);

                    raw_iteration._value = 0
                    // Object.defineProperty(raw_iteration, "value", {
                    //     get() {  
                    //         return this._value
                    //     },
                    //     set(value) {
                    //         this._value = value;
                    //     },
                    // });

                    const value_preview = ComfyWidgets["STRING"](this, "PREVIEW_raw_iteration", ["STRING", { multiline: true }], app).widget;
                    value_preview.inputEl.readOnly = true;
                    value_preview.inputEl.disabled = true;


                    // value_preview.inputEl.style.opacity = 0.6;
                    value_preview.inputEl.style.textAlign = "center";
                    value_preview.inputEl.style.fontSize = "2.5em";
                    value_preview.inputEl.style.backgroundColor = "black";

                    value_preview.inputEl.style.setProperty("--comfy-input-bg", "transparent");
                    value_preview.inputEl.style.setProperty("background", "red", "important");
                    // remove the comfy-multiline-input class

                    // disallow user selection
                    value_preview.inputEl.style.userSelect = "none";

                    const loop_preview = ComfyWidgets["STRING"](this, "PREVIEW_raw_iteration", ["STRING", { multiline: true }], app).widget;
                    loop_preview.inputEl.readOnly = true;
                    loop_preview.inputEl.disabled = true;


                    // loop_preview.inputEl.style.opacity = 0.6;
                    loop_preview.inputEl.style.textAlign = "center";
                    loop_preview.inputEl.style.fontSize = "1.5em";
                    loop_preview.inputEl.style.backgroundColor = "black";

                    loop_preview.inputEl.style.setProperty("--comfy-input-bg", "transparent");
                    loop_preview.inputEl.style.setProperty("background", "red", "important");
                    // remove the comfy-multiline-input class

                    // disallow user selection
                    loop_preview.inputEl.style.userSelect = "none";

                    const onReset = () => {
                        raw_iteration.value = 0;
                        raw_loop.value = 0;

                        value_preview.value = 0;
                        loop_preview.value = 0;

                        app.canvas.setDirty(true);
                    }

                    const reset_button = this.addWidget("button", `Reset`, "reset", onReset);

                    const run_button = this.addWidget("button", `Queue`, "queue", () => {
                        onReset() // this could maybe be a setting or checkbox
                        app.queuePrompt(0, total_frames.value * loop_count.value)

                    });



                    raw_iteration.afterQueued = function () {
                        this.value++;
                        raw_loop.value = Math.floor(this.value / total_frames.value);
                        value_preview.value = `raw: ${this.value}
frame: ${this.value % total_frames.value}`;
                        if (raw_loop.value + 1 > loop_count.value) {
                            loop_preview.value = `Done 😎!`
                            return
                        }

                        loop_preview.value = `current loop: ${raw_loop.value + 1}/${loop_count.value}`

                    }
                    // ComfyWidgets.hideWidget()
                    // if (raw_iteration > 0) {
                    // raw_iteration.inputEl.readOnly = true
                    // raw_iteration.inputEl.style.opacity = 0.6;
                    // console.log(raw_iteration)
                    // raw_iteration.onRemove?.();
                    // console.log(this.widgets[raw_iteration].type);

                    // raw_iteration.afterQueued = function () {
                    //     this.value++
                    // }
                    // }




                }
                //LiteGraph.ALWAYS
                const onExecuted = nodeType.prototype.onExecuted;

                nodeType.prototype.onExecuted = function (data) {
                    console.log("executed")
                    onExecuted?.apply(this, data)
                    if (this.widgets) {
                        const pos = this.widgets.findIndex((w) => w.name === "preview");
                        if (pos !== -1) {
                            for (let i = pos; i < this.widgets.length; i++) {
                                this.widgets[i].onRemove?.();
                            }
                            this.widgets.length = pos;
                        }
                    }

                    const w = ComfyWidgets["STRING"](this, "preview", ["STRING", { multiline: true }], app).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = data.total_frames;

                    // this.onResize?.(this.size);
                    this.setSize?.(this.computeSize())

                }
                // const onAfterExecuteNode = nodeType.prototype.onAfterExecuteNode;
                // nodeType.prototype.onAfterExecuteNode = function () {
                //     onAfterExecuteNode?.apply(this)
                //     console.log("after", this)

                // }
                console.debug(`Registered ${nodeType.name} node extra events`)
                break

            }

            case "Debug (mtb)": {
                // console.log(`Registered ${nodeType.name} node extra events`)
                // nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                //     // const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                //     console.debug({
                //         type, index, connected, link_info
                //     })

                //     shared.dynamic_connection(this, index, connected, "anything_", "*")
                // }

                // const onNodeCreated = nodeType.prototype.onNodeCreated;
                // nodeType.prototype.onNodeCreated = function () {
                //     const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                //     this.addInput(`input_${this.inputs.length + 1}`, "*")

                // }

                break

            }
            default: {
                break

            }

        }
        // const onNodeCreated = nodeType.prototype.onNodeCreated;

        // nodeType.prototype.onNodeCreated = function () {
        //     const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        // }




        // console.log(nodeData.output)
        // if (nodeData.output.includes("VIDEO") && nodeData.output_node) {
        //     console.log(`Found video output for ${nodeType}`)
        //     console.log(nodeData)

        // }

        // if (nodeData.name === "Psd Save (mtb)") {
        //     console.log(`Found psd node`)
        //     console.log(nodeData)

        // }



    }
};


app.registerExtension(mtb_widgets);