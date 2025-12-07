<script lang="ts">
  import { dndzone } from 'svelte-dnd-action'
  import InspectorItem from './InspectorItem.svelte'
  import { flip } from 'svelte/animate'
  import 'iconify-icon'

  interface InputItem {
    id: number
    [key: string]: unknown
  }

  let { inputs = [] as InputItem[] } = $props()

  const flipDurationMs = 100
  let locked = $state(false)

  function handleDndConsider(e: CustomEvent<{ items: InputItem[] }>) {
    inputs = e.detail.items
  }
  function handleDndFinalize(e: CustomEvent<{ items: InputItem[] }>) {
    inputs = e.detail.items
  }
</script>

<div id="controls">
  <button id="lock" onclick={() => (locked = !locked)}>
    {#if locked}
      <iconify-icon icon="uis:lock"></iconify-icon>
    {:else}
      <iconify-icon icon="uis:unlock"></iconify-icon>
    {/if}
  </button>
  <section
    use:dndzone={{ items: inputs, flipDurationMs, dragDisabled: locked }}
    onconsider={handleDndConsider}
    onfinalize={handleDndFinalize}
  >
    {#each inputs as item, index (item.id)}
      <div animate:flip={{ duration: flipDurationMs }}>
        <InspectorItem {item} />
      </div>
    {/each}
  </section>
</div>

<style>
  #lock {
    font-size: 2em;
    background: none;
    border: none;
  }
  section {
    width: 100%;
    /* padding: 0.3em; */
    overflow: auto;
  }
  div {
    margin: 0.15em 0;
  }
</style>
