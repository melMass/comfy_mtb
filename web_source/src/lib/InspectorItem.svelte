<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import InspectorInput from './InspectorInput.svelte'
  import { inComfy } from './utils.js'

  interface ItemType {
    name: string
    type: string
    node_id?: number
    id: number
    value?: unknown
    widgets?: { value: unknown }[]
    options?: string[]
    [key: string]: unknown
  }

  interface Action {
    label: string
    icon?: string
    callback: () => void
  }

  let {
    item = {
      name: 'KSampler',
      type: 'STRING',
      node_id: 123,
      id: 4,
    } as ItemType,
    extra_actions = {} as Record<string, Action>,
  } = $props()

  let actions = $state<Record<string, Action>>({})

  // Derive value from widget - updates when widget value changes
  const getWidgetValue = () => item.widgets?.[0]?.value ?? item.value ?? ''
  let value = $state<unknown>(getWidgetValue())

  // Sync value from widget when it changes (Node → Panel)
  // Use untrack to prevent reacting to local value changes (avoids reverting user input)
  $effect(() => {
    const widgetVal = getWidgetValue()
    untrack(() => {
      if (widgetVal !== value) {
        value = widgetVal
      }
    })
  })

  onMount(() => {
    actions = {
      goToNode: {
        label: 'Focus',
        icon: '↗',
        callback: () => {
          const id = item.node_id
          if (!id) return
          if (!inComfy()) return
          // @ts-expect-error - app is global in ComfyUI
          const app = window.app
          const node = app.graph.getNodeById(id)
          app.canvas.centerOnNode(node)
          app.canvas.setZoom(1)
          app.canvas.selectNode(node)
        },
      },
      ...extra_actions,
    }
    if (item.name.toLowerCase() === 'seed') {
      actions.randomize = {
        label: 'Random',
        icon: '⚄',
        callback: () => {
          value = Math.floor(Math.random() * 1e9)
          if (!inComfy()) return
          onInput(null, value)
          // @ts-expect-error - app is global in ComfyUI
          window.app.canvas.setDirty(true)
        },
      }
    }
  })

  const onInput = (e: Event | null, val?: unknown) => {
    if (!inComfy()) return
    if (item.widgets) {
      for (const w of item.widgets as Array<{
        value: unknown
        callback?: (value: unknown) => void
        options?: { callback?: (value: unknown) => void }
      }>) {
        const newVal = val ?? (e?.target as HTMLInputElement)?.value
        w.value = newVal
        // Trigger widget callback to notify ComfyUI of the change
        w.callback?.(newVal)
        w.options?.callback?.(newVal)
      }
      // @ts-expect-error - app is global in ComfyUI
      window.app.canvas.setDirty(true)
    }
  }

  const typeLabels: Record<string, string> = {
    STRING: 'Text',
    NUMBER: 'Number',
    COMBO: 'Select',
    MODEL: 'Model',
    BOOLEAN: 'Toggle',
    IMAGE: 'Image',
    COLOR: 'Color',
  }
</script>

<div class="mtb-input-row">
  <div class="mtb-input-header">
    <label class="mtb-input-label" for="input-{item.id}">{item.name}</label>
    <span class="mtb-input-type">{typeLabels[item.type] || item.type}</span>
  </div>

  <div class="mtb-input-field">
    <InspectorInput {onInput} bind:value id={item.id} {item} />
  </div>

  {#if Object.keys(actions).length > 0}
    <div class="mtb-input-actions">
      {#each Object.keys(actions) as k}
        {@const action = actions[k]}
        <button
          class="mtb-action-btn"
          onclick={action.callback}
          title={action.label}
        >
          {#if action.icon}
            <span class="mtb-action-icon">{action.icon}</span>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .mtb-input-row {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.06));
    transition: background 0.15s ease;
  }

  .mtb-input-row:hover {
    background: rgba(255,255,255,0.02);
  }

  .mtb-input-row:last-child {
    border-bottom: none;
  }

  .mtb-input-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .mtb-input-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--fg-color, #fafafa);
    letter-spacing: -0.01em;
  }

  .mtb-input-type {
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--descrip-text, rgba(255,255,255,0.4));
    padding: 2px 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
  }

  .mtb-input-field {
    display: flex;
    gap: 8px;
  }

  .mtb-input-actions {
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }

  .mtb-action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    padding: 0;
    background: transparent;
    border: 1px solid var(--border-color, rgba(255,255,255,0.1));
    border-radius: 4px;
    color: var(--descrip-text, rgba(255,255,255,0.5));
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .mtb-action-btn:hover {
    background: var(--mtb-api-color, #2930b0);
    border-color: var(--mtb-api-color, #2930b0);
    color: white;
    transform: translateY(-1px);
  }

  .mtb-action-icon {
    font-size: 12px;
  }
</style>
