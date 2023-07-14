// Define the Color Picker widget class
import parseCss from '/extensions/mtb/extern/parse-css.js'
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

export function CUSTOM_INT(node, inputName, val, func, config = {}) {
    return {
        widget: node.addWidget(
            "number",
            inputName,
            val,
            func,
            Object.assign({}, { min: 0, max: 4096, step: 640, precision: 0 }, config)
        ),
    };
}
const dumb_call = (v, d, node) => {
    console.log("dumb_call", { v, d, node });
}
function isColorBright(rgb, threshold = 240) {
    const brightess = getBrightness(rgb)

    return brightess > threshold
}

function getBrightness(rgbObj) {
    return Math.round(((parseInt(rgbObj[0]) * 299) + (parseInt(rgbObj[1]) * 587) + (parseInt(rgbObj[2]) * 114)) / 1000)
}


/**
 * @returns {import("./types/litegraph").IWidget} widget
 */
const custom = (key, val, compute = false) => {
    /** @type {import("/types/litegraph").IWidget} */
    const widget = {}
    // widget.y = 0;
    widget.name = key;
    widget.type = "COLOR";
    widget.options = { default: "#ff0000" };
    widget.value = val || "#ff0000";
    widget.draw = function (ctx,
        node,
        widgetWidth,
        widgetY,
        height) {
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
        ctx.fillStyle = isColorBright(color.values, 125) ? "#000" : "#fff";


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
                    // picker.style.position = "absolute";
                    // picker.style.left = (window.innerWidth / 2) + "px";
                    // picker.style.top = (window.innerHeight / 2) + "px";
                    // picker.style.transform = "translate(-50%, -50%)";
                    // picker.style.zIndex = 1000;



                    document.body.appendChild(picker);

                    picker.addEventListener("change", () => {
                        this.value = picker.value;
                        node.graph._version++;
                        node.setDirtyCanvas(true, true);
                        document.body.removeChild(picker);
                    });

                    // simulate click with screen center
                    const pointer_event = new MouseEvent('click', {
                        bubbles: false,
                        // cancelable: true,
                        pointerType: "mouse",
                        clientX: window.innerWidth / 2,
                        clientY: window.innerHeight / 2,
                        x: window.innerWidth / 2,
                        y: window.innerHeight / 2,
                        offsetX: window.innerWidth / 2,
                        offsetY: window.innerHeight / 2,
                        screenX: window.innerWidth / 2,
                        screenY: window.innerHeight / 2,


                    });
                    console.log(e)
                    picker.dispatchEvent(pointer_event);

                }
            }
        }
    }
    widget.computeSize = function (width) {
        return [width, 32];
    }

    return widget;
}

app.registerExtension({
    name: "mtb.ColorPicker",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        //console.log("mtb.ColorPicker", { nodeType, nodeData, app });
        const rinputs = nodeData.input?.required; // object with key/value pairs, "0" is the type
        // console.log(nodeData.name, { nodeType, nodeData, app });

        if (!rinputs) return;


        let has_color = false;
        for (const [key, input] of Object.entries(rinputs)) {
            if (input[0] === "COLOR") {
                has_color = true;
                // input[1] = { default: "#ff0000" };

            }
        }

        if (!has_color) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.serialize_widgets = true;
            // if (rinputs[0] === "COLOR") {
            // console.log(nodeData.name, { nodeType, nodeData, app });

            // loop through the inputs to find the color inputs
            for (const [key, input] of Object.entries(rinputs)) {
                if (input[0] === "COLOR") {
                    let widget = custom(key, input[1])

                    this.addCustomWidget(widget)
                }
                // }
            }

            this.onRemoved = function () {
                // When removing this node we need to remove the input from the DOM
                for (let y in this.widgets) {
                    if (this.widgets[y].canvas) {
                        this.widgets[y].canvas.remove();
                    }
                }
            };
        }
    }
});
