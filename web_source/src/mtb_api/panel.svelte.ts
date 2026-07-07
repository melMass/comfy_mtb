/**
 * API Panel Controller
 * Manages the Svelte-based API panel for controlling exposed inputs
 *
 * NOTE: This file uses .svelte.ts extension to enable Svelte 5 runes ($state)
 */

import { mount, unmount } from 'svelte'
import * as shared from '@mtb/shared'
import Inspector from '../lib/Inspector.svelte'
import type { APIInput, MTBNode } from './types'

interface InputItem {
  id: number
  name: string
  type: string
  options?: string[]
  [key: string]: unknown
}

/**
 * Reactive props state for the Inspector component
 * Using $state makes this reactive - changes auto-update the component
 */
function createPanelProps() {
  let props = $state({
    visible: true,
    inputs: {} as Record<string, InputItem>,
  })
  return props
}

/** Custom event for API changes */
export const MTB_API_CHANGED_EVENT = 'mtb:api:changed'

/** Custom event for order changes from drag-drop */
export const MTB_API_ORDER_CHANGED_EVENT = 'mtb:api:order-changed'

/** Dispatch event to notify panel of changes */
export function notifyAPIChanged(): void {
  window.dispatchEvent(new CustomEvent(MTB_API_CHANGED_EVENT))
}

/** Dispatch event with new order after drag-drop */
export function notifyOrderChanged(orderedInputs: { node_id: number; original_name: string; order: number }[]): void {
  window.dispatchEvent(new CustomEvent(MTB_API_ORDER_CHANGED_EVENT, { detail: orderedInputs }))
}

/**
 * Controls the API panel UI for managing exposed workflow inputs
 */
export class APIPanel {
  private component: ReturnType<typeof mount> | null = null
  private props = createPanelProps()
  private sidebarMode = false
  private listening = false

  constructor() {
    // Don't mount automatically - wait for renderInto or show
    this.setupChangeListener()
  }

  /**
   * Listen for API changes and update panel reactively
   */
  private setupChangeListener(): void {
    if (this.listening) return
    this.listening = true

    window.addEventListener(MTB_API_CHANGED_EVENT, () => {
      // Debounce updates slightly
      requestAnimationFrame(() => {
        this.updateContent()
      })
    })

    // Listen for order changes from drag-drop
    window.addEventListener(MTB_API_ORDER_CHANGED_EVENT, ((e: CustomEvent<{ node_id: number; original_name: string; order: number }[]>) => {
      this.applyInputOrder(e.detail)
    }) as EventListener)
  }

  /**
   * Applies new order to node properties after drag-drop reorder
   */
  private applyInputOrder(orderedInputs: { node_id: number; original_name: string; order: number }[]): void {
    for (const node of shared.getNodes(true) as MTBNode[]) {
      const nodeInputs = orderedInputs.filter(i => i.node_id === node.id)
      if (nodeInputs.length === 0) continue

      for (const input of nodeInputs) {
        if (node.properties.mtb_api?.inputs?.[input.original_name]) {
          node.properties.mtb_api.inputs[input.original_name].order = input.order
        }
      }
      // Trigger property update
      node.setProperty('mtb_api', node.properties.mtb_api)
    }
  }

  /**
   * Creates and mounts the Inspector Svelte component
   */
  private createPanel(target: HTMLElement = document.body) {
    return mount(Inspector, {
      target,
      props: this.props,
    })
  }

  /**
   * Renders the panel into a sidebar element
   */
  renderInto(el: HTMLElement): void {
    this.sidebarMode = true
    this.props.visible = true

    // Destroy existing component if any
    if (this.component) {
      unmount(this.component)
    }

    this.component = this.createPanel(el)
    this.updateContent()
  }

  /**
   * Shows the panel and updates its content
   */
  show(): void {
    if (!this.component) {
      this.component = this.createPanel()
    }
    this.updateContent()
    this.props.visible = true
  }

  /**
   * Hides the panel
   */
  hide(): void {
    this.props.visible = false
  }

  /**
   * Returns whether the panel is currently visible
   */
  isVisible(): boolean {
    return this.props.visible
  }

  /**
   * Toggles panel visibility
   */
  toggle(): void {
    if (this.isVisible()) {
      this.hide()
    } else {
      this.show()
    }
  }

  /**
   * Collects all API inputs from marked nodes in the graph
   */
  getAPIInputs(): Record<string, APIInput> {
    const inputsList: (APIInput & { original_name: string })[] = []

    for (const node of shared.getNodes(true) as MTBNode[]) {
      const widgets = node.widgets

      if (node.properties.mtb_api && node.properties.useAPI) {
        if (node.properties.mtb_api.inputs) {
          for (const currentName in node.properties.mtb_api.inputs) {
            const current = node.properties.mtb_api.inputs[currentName]
            if (current.enabled) {
              const inputName = current.name || currentName
              const widget = widgets?.find((w) => w.name === currentName)
              if (!widget) continue

              inputsList.push({
                ...current,
                id: 0, // Will be assigned after sorting
                name: inputName,
                original_name: currentName,
                type: current.type,
                node_id: node.id,
                widgets: [widget],
                // Extract current value from widget
                value: widget.value,
                // For COMBO types, extract options
                options: (widget.options as { values?: string[] })?.values,
              })
            }
          }
        }
      }
    }

    // Sort by order (undefined order goes to end)
    inputsList.sort((a, b) => {
      const orderA = a.order ?? Number.MAX_SAFE_INTEGER
      const orderB = b.order ?? Number.MAX_SAFE_INTEGER
      return orderA - orderB
    })

    // Convert to Record and assign sequential IDs
    const inputs: Record<string, APIInput> = {}
    inputsList.forEach((input, index) => {
      input.id = index + 1
      inputs[input.name] = input
    })

    return inputs
  }

  /**
   * Updates the panel content with current API inputs
   */
  updateContent(): void {
    const newInputs = this.getAPIInputs()
    // Update the reactive props - this auto-updates the component
    this.props.inputs = newInputs
    console.log('Found API inputs:', newInputs)
  }
}

/** Singleton panel instance */
let panelInstance: APIPanel | null = null

/**
 * Gets or creates the singleton panel instance
 */
export function getAPIPanel(): APIPanel {
  if (!panelInstance) {
    panelInstance = new APIPanel()
  }
  return panelInstance
}
