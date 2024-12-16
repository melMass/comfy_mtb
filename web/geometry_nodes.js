/**
 * File: geometry_nodes.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

app.registerExtension({
  name: 'mtb.geometry_nodes',
  init: () => {},

  async beforeRegisterNodeDef(nodeType, nodeData, ...args) {
    switch (nodeData.name) {
      case 'Geometry Load (mtb)': {
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          onExecuted?.apply(this, nodeType, nodeData, ...args)
          console.log('Executed Load Geometry', ...args)
          console.log('Message:', message)
        }
        break
      }
    }
  },
})
