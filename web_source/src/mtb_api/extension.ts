/**
 * MTB API Extension Registration
 * Registers the API authoring layer with ComfyUI
 */

import { app } from '@/scripts/app'
import * as shared from '@mtb/shared'
import { apiSettingsWidgetManager } from './widget-manager'
import { getAPIPanel } from './panel.svelte'
import type { MTBNode } from './types'

interface ContextMenuItem {
  content: string
  callback: (...args: unknown[]) => void
}

interface NodeType {
  prototype: {
    onDrawForeground?: (...args: unknown[]) => void
  }
}

/**
 * Registers the MTB API extension with ComfyUI
 */
export function registerMtbApiExtension(): void {
  app.registerExtension({
    name: 'mtb.api',

    setup() {
      // Keyboard shortcut: Ctrl+Shift+A
      const panel = getAPIPanel()
      document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'A') {
          e.preventDefault()
          panel.toggle()
        }
      })
    },

    init() {
      // Register sidebar tab
      // @ts-expect-error - extensionManager is ComfyUI API
      app.extensionManager.registerSidebarTab({
        id: 'mtb-api-panel',
        icon: 'pi pi-bolt',
        title: 'API Authoring',
        tooltip: 'MTB: Configure API inputs and outputs',
        type: 'custom',
        render: (el: HTMLElement) => {
          const panel = getAPIPanel()
          panel.renderInto(el)
        },
        destroy: () => {
          // Cleanup if needed
        },
      })
    },

    async beforeRegisterNodeDef(
      nodeType: NodeType,
      _nodeData: unknown,
      _app: unknown,
    ) {
      // Add API menu to all nodes
      shared.addMenuHandler(nodeType, function (
        this: MTBNode,
        _app: unknown,
        options: ContextMenuItem[],
      ) {
        // Mark/Unmark as API input
        const markApiItem: ContextMenuItem = {
          content: this.properties.useAPI ? 'Unmark API ⚡' : 'Mark API ⚡',
          callback: (...args: unknown[]) => {
            const node = args[4] as MTBNode
            if (node.properties.useAPI) {
              node.setProperty('useAPI', false)
              apiSettingsWidgetManager.ensureWidgets(node)
            } else {
              node.setProperty('useAPI', true)
            }
          },
        }
        options.push(markApiItem)

        // Mark/Remove as API output
        const markOutputItem: ContextMenuItem = {
          content: this.properties.mtb_api?.isAPIOutput
            ? 'Remove Output (API) ⚡'
            : 'Mark Output (API) ⚡',
          callback: (...args: unknown[]) => {
            const node = args[4] as MTBNode
            if (node.properties.mtb_api?.isAPIOutput) {
              apiSettingsWidgetManager.applySettings(node, { isAPIOutput: false })
              apiSettingsWidgetManager.ensureWidgets(node)
            } else {
              apiSettingsWidgetManager.applySettings(node, { isAPIOutput: true })
              node.setProperty('useAPI', true)
            }
          },
        }
        options.push(markOutputItem)

        // Show/Hide disabled inputs
        const currentShow =
          this.properties.mtb_api?.showDisabled === undefined
            ? true
            : this.properties.mtb_api?.showDisabled

        const hideDisabledItem: ContextMenuItem = {
          content: currentShow
            ? 'Hide Disabled (API) ⚡'
            : 'Show Disabled (API) ⚡',
          callback: (...args: unknown[]) => {
            const node = args[4] as MTBNode
            const oldApi = node.properties.mtb_api || {}
            if (oldApi?.showDisabled !== undefined) {
              node.setProperty('mtb_api', {
                ...oldApi,
                showDisabled: !oldApi.showDisabled,
              })
            } else {
              node.setProperty('mtb_api', {
                ...oldApi,
                showDisabled: false,
              })
            }
            apiSettingsWidgetManager.ensureWidgets(node, true)
          },
        }
        options.push(hideDisabledItem)

        return [markApiItem, hideDisabledItem]
      })

      // Extend onDrawForeground to draw API indicators
      const origDrawForeground = nodeType.prototype.onDrawForeground
      nodeType.prototype.onDrawForeground = function (
        this: MTBNode,
        ctx: CanvasRenderingContext2D,
        canvas: unknown,
      ) {
        origDrawForeground?.apply(this, [ctx, canvas])
        apiSettingsWidgetManager.drawForeground(this, ctx, canvas)
      }
    },
  })
}
