/**
 * API Panel Controller
 * Manages the Svelte-based API panel for controlling exposed inputs
 */

import { mount } from 'svelte'
import * as shared from '@mtb/shared'
import Inspector from '../lib/Inspector.svelte'
import type { APIInput, MTBNode } from './types'

/** Svelte component with internal $$set and $$ context */
interface SvelteComponent {
  $$set: (props: Record<string, unknown>) => void
  $$: { ctx: unknown[] }
}

/**
 * Controls the API panel UI for managing exposed workflow inputs
 */
export class APIPanel {
  private panel: SvelteComponent
  private inputs: Record<string, APIInput> = {}

  constructor() {
    this.panel = this.createPanel()
  }

  /**
   * Creates and mounts the Inspector Svelte component
   */
  private createPanel(): SvelteComponent {
    const panel = mount(Inspector, {
      target: document.body,
      props: { visible: false },
    }) as unknown as SvelteComponent
    return panel
  }

  /**
   * Shows the panel and updates its content
   */
  show(): void {
    this.updateContent()
    this.panel.$$set({ visible: true })
  }

  /**
   * Hides the panel
   */
  hide(): void {
    this.panel.$$set({ visible: false })
  }

  /**
   * Returns whether the panel is currently visible
   */
  isVisible(): boolean {
    return this.panel.$$.ctx[0] === true
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
    const inputs: Record<string, APIInput> = {}
    let counter = 1

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

              if (!(inputName in inputs)) {
                inputs[inputName] = {
                  ...current,
                  id: counter,
                  name: inputName,
                  type: current.type,
                  node_id: node.id,
                  widgets: [],
                }
              }
              inputs[inputName].widgets.push(widget)
              counter++
            }
          }
        }
      }
    }

    return inputs
  }

  /**
   * Updates the panel content with current API inputs
   */
  updateContent(): void {
    this.inputs = this.getAPIInputs()
    this.panel.$$set({ inputs: this.inputs })
    console.log('Found API inputs:', this.inputs)
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
