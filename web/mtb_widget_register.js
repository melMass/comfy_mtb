import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const custom = (node, inputName, inputData, app) => {


    // nodeData.input.required.upload = ["IMAGEUPLOAD"];
    const imageWidget = node.widgets.find((w) => w.name === "image");
    let previewWidget;
    function showImage(name) {
        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            app.graph.setDirtyCanvas(true);
        };
        let folder_separator = name.lastIndexOf("/");
        let subfolder = "";
        if (folder_separator > -1) {
            subfolder = name.substring(0, folder_separator);
            name = name.substring(folder_separator + 1);
        }
        img.src = `/view?filename=${name}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`;
        node.setSizeForImage?.();
    }
    var default_value = imageWidget.value;
    Object.defineProperty(imageWidget, "value", {
        set: function (value) {
            this._real_value = value;
        },

        get: function () {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if (real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });
    // Add our own callback to the combo widget to render an image when it changes
    const cb = node.callback;
    imageWidget.callback = function () {
        showImage(imageWidget.value);
        if (cb) {
            return cb.apply(this, arguments);
        }
    };
    // On load if we have a value then render the image
    // The value isnt set immediately so we need to wait a moment
    // No change callbacks seem to be fired on initial setting of the value
    requestAnimationFrame(() => {
        if (imageWidget.value) {
            showImage(imageWidget.value);
        }
    });

    return { widget: previewWidget };



}

const dynamic_connection = (node, index, connected, connectionPrefix = "input_", connectionType = "PSDLAYER") => {

    // remove all non connected inputs
    if (!connected && node.inputs.length > 1) {
        node.removeInput(index)

        // make inputs sequential again
        for (let i = 0; i < node.inputs.length; i++) {
            node.inputs[i].name = `${connectionPrefix}${i + 1}`
        }
    }

    // add an extra input
    if (node.inputs[node.inputs.length - 1].link != undefined) {
        node.addInput(`${connectionPrefix}${node.inputs.length + 1}`, connectionType)
    }


}

/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const mtb_widgets = {
    name: "mtb.core.register",

    getCustomWidgets: async (app) => {
        return {
            VIDEO: (node, inputName, inputData, app) => {
                const videoWidget = {
                    name: "VideoWidget",
                    description: "Video Player Widget",
                    value: inputData,
                    properties: {},
                    widget: null,

                    init: function () {
                        this.widget = document.createElement("video");
                        this.widget.width = 200;
                        this.widget.height = 120;
                        this.widget.controls = true;
                        this.widget.style.width = "100%";
                        this.widget.style.height = "100%";
                        this.widget.style.objectFit = "contain";
                        this.widget.style.backgroundColor = "black";
                        this.widget.style.pointerEvents = "none";
                        node.addWidget(inputName, videoWidget.widget, inputData);
                    },

                    setValue: function (value, options) {
                        if (value instanceof HTMLVideoElement) {
                            this.widget.src = value.src;
                        } else if (typeof value === "string") {
                            this.widget.src = value;
                        }
                    },

                    getValue: function () {
                        return this.widget.src;
                    },

                    append: function (parent) {
                        parent.appendChild(this.widget);
                    },

                    remove: function () {
                        this.widget.parentNode.removeChild(this.widget);
                    }
                };
                return {
                    widget: videoWidget,
                }
            }
        }
    },
    // init: () => {
    //     // Define the custom widget
    //     var videoWidget = {
    //         name: "VideoWidget",
    //         description: "Video Player Widget",
    //         value: null,
    //         properties: {},
    //         widget: null,

    //         // Initialization function
    //         init: function () {
    //             this.widget = document.createElement("video");
    //             this.widget.width = 200;
    //             this.widget.height = 120;
    //             this.widget.controls = true;
    //             this.widget.style.width = "100%";
    //             this.widget.style.height = "100%";
    //             this.widget.style.objectFit = "contain";
    //             this.widget.style.backgroundColor = "black";
    //             this.widget.style.pointerEvents = "none";
    //         },

    //         // Set the value function
    //         setValue: function (value, options) {
    //             if (value instanceof HTMLVideoElement) {
    //                 this.widget.src = value.src;
    //             } else if (typeof value === "string") {
    //                 this.widget.src = value;
    //             }
    //         },

    //         // Get the value function
    //         getValue: function () {
    //             return this.widget.src;
    //         },

    //         // Append the widget to the parent element
    //         append: function (parent) {
    //             parent.appendChild(this.widget);
    //         },

    //         // Remove the widget from the parent element
    //         remove: function () {
    //             this.widget.parentNode.removeChild(this.widget);
    //         }
    //     };

    //     // Register the custom widget in LiteGraph.js
    //     LiteGraph.registered_widgets["VIDEO"] = videoWidget;
    // },


    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Psd Save (mtb)") {

            // const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                // const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                dynamic_connection(this, index, connected)
                }


            }
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