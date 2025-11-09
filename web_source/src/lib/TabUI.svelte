<script lang="ts">
  import { dndzone } from 'svelte-dnd-action'
  import InspectorItem from './InspectorItem.svelte'
  import { flip } from 'svelte/animate'
  import 'iconify-icon'
  export let inputs = []
  const flipDurationMs = 100
  function handleDndConsider(e) {
    inputs = e.detail.items
  }
  function handleDndFinalize(e) {
    inputs = e.detail.items
  }
  let locked = false
</script>

<div id="controls">
  <button id="lock" on:click={() => (locked = !locked)}>
    {#if locked}
      <iconify-icon icon="uis:lock"></iconify-icon>
    {:else}
      <iconify-icon icon="uis:unlock"></iconify-icon>
    {/if}
  </button>
  <section
    use:dndzone={{ items: inputs, flipDurationMs, dragDisabled: locked }}
    on:consider={handleDndConsider}
    on:finalize={handleDndFinalize}
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
