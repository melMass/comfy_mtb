<script lang="ts">
  import { onMount } from 'svelte'
  import InspectorInput from './InspectorInput.svelte'
  import { inComfy } from './utils.js'

  interface ItemType {
    name: string
    type: string
    node_id?: number
    id: number
    widgets?: { value: unknown }[]
    [key: string]: unknown
  }

  interface Action {
    label: string
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
  let value = $state<unknown>(undefined)

  onMount(() => {
    actions = {
      goToNode: {
        label: '⬆',
        callback: () => {
          const id = item.node_id
          if (!id) {
            console.log('No nodeid')
            return
          }
          if (!inComfy()) {
            console.log('NOT IN COMFY')
            return
          }
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
        label: '🎲',
        callback: () => {
          value = Math.floor(Math.random() * 1e9)
          if (!inComfy()) {
            console.log('NOT IN COMFY')
            return
          }
          onInput(null, value)
          // @ts-expect-error - app is global in ComfyUI
          window.app.canvas.setDirty(true)
        },
      }
    }
    if (item.widgets) {
      value = item.widgets[0].value
    }
  })

  const onInput = (e: Event | null, val?: unknown) => {
    if (!inComfy()) {
      console.log(`Not in comfy ${(e?.target as HTMLInputElement)?.value}`)
      return
    }
    if (item.widgets) {
      for (let i = 0; i < item.widgets.length; i++) {
        const w = item.widgets[i]
        w.value = val ?? (e?.target as HTMLInputElement)?.value
      }
      // @ts-expect-error - app is global in ComfyUI
      window.app.canvas.setDirty(true)
    }
  }
</script>

<div class="container">
  <label for="input">{item.name}</label>
  <div id="input">
    <InspectorInput {onInput} bind:value id={item.id} {item} />
  </div>
  <div id="buttons">
    {#each Object.keys(actions) as k}
      {@const action = actions[k]}
      <button onclick={action.callback}>{action.label}</button>
    {/each}
  </div>
</div>

<style>
  .container {
    text-align: center;
    padding: 0.5em 0em;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
</style>
