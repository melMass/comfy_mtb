import { app } from "/scripts/app.js";

//- WIDGET UTILS
const CONVERTED_TYPE = "converted-widget";

export function offsetDOMWidget(widget, ctx, node, widgetWidth, widgetY, height) {
    const margin = 10;
    const elRect = ctx.canvas.getBoundingClientRect();
    const transform = new DOMMatrix()
        .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
        .multiplySelf(ctx.getTransform())
        .translateSelf(margin, margin + widgetY);

    const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
    Object.assign(widget.inputEl.style, {
        transformOrigin: "0 0",
        transform: scale,
        left: `${transform.a + transform.e}px`,
        top: `${transform.d + transform.f}px`,
        width: `${widgetWidth - (margin * 2)}px`,
        height: `${widget.parent.inputHeight - (margin * 2)}px`,
        position: "absolute",
        background: (!node.color) ? '' : node.color,
        color: (!node.color) ? '' : 'white',
        zIndex: app.graph._nodes.indexOf(node),
    })
}

/**
 * Extracts the type and link type from a widget config object.
 * @param {*} config 
 * @returns 
 */
export function getWidgetType(config) {
    // Special handling for COMBO so we restrict links based on the entries
    let type = config[0];
    let linkType = type;
    if (type instanceof Array) {
        type = "COMBO";
        linkType = linkType.join(",");
    }
    return { type, linkType };
}

export const dynamic_connection = (node, index, connected, connectionPrefix = "input_", connectionType = "PSDLAYER") => {

    // remove all non connected inputs
    if (!connected && node.inputs.length > 1) {
        console.debug(`Removing input ${index} (${node.inputs[index].name})`)
        if (node.widgets) {
            const w = node.widgets.find((w) => w.name === node.inputs[index].name);
            if (w) {

                w.onRemove?.();
                node.widgets.length = node.widgets.length - 1
            }
        }
        node.removeInput(index)

        // make inputs sequential again
        for (let i = 0; i < node.inputs.length; i++) {
            node.inputs[i].name = `${connectionPrefix}${i + 1}`
        }
    }

    // add an extra input
    if (node.inputs[node.inputs.length - 1].link != undefined) {
        console.debug(`Adding input ${node.inputs.length + 1} (${connectionPrefix}${node.inputs.length + 1})`)

        node.addInput(`${connectionPrefix}${node.inputs.length + 1}`, connectionType)
    }

}


/**
 * Appends a callback to the extra menu options of a given node type.
 * @param {*} nodeType 
 * @param {*} cb 
 */
export function addMenuHandler(nodeType, cb) {
    const getOpts = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = getOpts.apply(this, arguments);
        cb.apply(this, arguments);
        return r;
    };
}

export function hideWidget(node, widget, suffix = "") {
    widget.origType = widget.type;
    widget.hidden = true
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;
    widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically
    widget.type = CONVERTED_TYPE + suffix;
    widget.serializeValue = () => {
        // Prevent serializing the widget if we have no input linked
        const { link } = node.inputs.find((i) => i.widget?.name === widget.name);
        if (link == null) {
            return undefined;
        }
        return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
    };

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidget(node, w, ":" + widget.name);
        }
    }
}

export function showWidget(widget) {
    widget.type = widget.origType;
    widget.computeSize = widget.origComputeSize;
    widget.serializeValue = widget.origSerializeValue;

    delete widget.origType;
    delete widget.origComputeSize;
    delete widget.origSerializeValue;

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            showWidget(w);
        }
    }
}

export function convertToWidget(node, widget) {
    showWidget(widget);
    const sz = node.size;
    node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name));

    for (const widget of node.widgets) {
        widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT;
    }

    // Restore original size but grow if needed
    node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}


export function convertToInput(node, widget, config) {
    hideWidget(node, widget);

    const { linkType } = shared.getWidgetType(config);

    // Add input and store widget config for creating on primitive node
    const sz = node.size;
    node.addInput(widget.name, linkType, {
        widget: { name: widget.name, config },
    });

    for (const widget of node.widgets) {
        widget.last_y += LiteGraph.NODE_SLOT_HEIGHT;
    }

    // Restore original size but grow if needed
    node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}

export function hideWidgetForGood(node, widget, suffix = "") {
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;
    widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically
    widget.type = CONVERTED_TYPE + suffix;
    // widget.serializeValue = () => {
    //     // Prevent serializing the widget if we have no input linked
    //     const w = node.inputs?.find((i) => i.widget?.name === widget.name);
    //     if (w?.link == null) {
    //         return undefined;
    //     }
    //     return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
    // };

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ":" + widget.name);
        }
    }
}


//- COLOR UTILS
export function isColorBright(rgb, threshold = 240) {
    const brightess = getBrightness(rgb)
    return brightess > threshold
}

function getBrightness(rgbObj) {
    return Math.round(((parseInt(rgbObj[0]) * 299) + (parseInt(rgbObj[1]) * 587) + (parseInt(rgbObj[2]) * 114)) / 1000)
}
