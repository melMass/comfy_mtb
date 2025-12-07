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

/**
 * Controls the API panel UI for managing exposed workflow inputs
 */
export class APIPanel {
  private component: ReturnType<typeof mount> | null = null
  private props = createPanelProps()
  private sidebarMode = false

  constructor() {
    // Don't mount automatically - wait for renderInto or show
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
                  // Extract current value from widget
                  value: widget.value,
                  // For COMBO types, extract options
                  options: (widget.options as { values?: string[] })?.values,
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
