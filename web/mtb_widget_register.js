import { app } from "/scripts/app.js";
/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const mtb_widgets = {
    name: "mtb.core.register",

    /**
     * 
     * @param {import("./types/litegraph").LGraphNode} node 
     */
    async nodeCreated(node, app) {
        if (node.comfyClass === "Psd Save (mtb)") {
            node.onConnectionsChange = function (type, index, connected, link_info) {

                // remove all non connected inputs
                if (!connected && node.inputs.length > 1) {
                    node.removeInput(index)

                    // make inputs sequential again
                    for (let i = 0; i < node.inputs.length; i++) {
                        node.inputs[i].name = `input_${i + 1}`
                    }
                }

                // add an extra input
                if (node.inputs[node.inputs.length - 1].link != undefined) {
                    node.addInput(`input_${node.inputs.length + 1}`, "PSDLAYER")
                }


            }

        }
    },

};


app.registerExtension(mtb_widgets);