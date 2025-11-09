<script lang="ts">
  import { onMount } from 'svelte'
  import InspectorInput from './InspectorInput.svelte'
  import { inComfy } from './utils.js'

  export let item = {
    name: 'KSampler',
    type: 'STRING',
    node_id: 123,
    id: 4,
  }
  export let extra_actions = {}
  let actions = {}
  let value

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
          // @ts-ignore
          const app = window.app
          const node = app.graph.getNodeById(id)
          app.canvas.centerOnNode(node)
          app.canvas.setZoom(1)
          app.canvas.selectNode(node)
        },
      },
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
          // element.dispatchEvent(new Event('input'))
          onInput(null, value)
          const app = window.app
          app.canvas.setDirty(true)
        },
      }
    }
    if (item.widgets) {
      value = item.widgets[0].value
    }
  })

  const onInput = (e, val) => {
    if (!inComfy()) {
      console.log(`Not in comfy ${e.target.value}`)
      return
    }
    if (item.widgets) {
      console.log(item.widgets)
      for (let i = 0; i < item.widgets.length; i++) {
        const w = item.widgets[i]
        w.value = val || e.target.value //value // control.value
      }
      app.canvas.setDirty(true)
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
      <button on:click={action.callback}>{action.label}</button>
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
