<script lang="ts">
  import { dndzone } from 'svelte-dnd-action'
  import InspectorItem from './InspectorItem.svelte'
  import { flip } from 'svelte/animate'

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

<div class="mtb-tab-ui">
  <div class="mtb-tab-toolbar">
    <button
      class="mtb-toolbar-btn"
      class:active={locked}
      onclick={() => (locked = !locked)}
      title={locked ? 'Unlock reordering' : 'Lock reordering'}
    >
      {#if locked}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
        </svg>
      {:else}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
          <path d="M7 11V7a5 5 0 0 1 9.9-1"></path>
        </svg>
      {/if}
    </button>
    <span class="mtb-toolbar-label">{inputs.length} input{inputs.length !== 1 ? 's' : ''}</span>
  </div>

  {#if inputs.length === 0}
    <div class="mtb-empty-state">
      <div class="mtb-empty-icon">⚡</div>
      <p class="mtb-empty-text">No API inputs</p>
      <p class="mtb-empty-hint">Right-click a node and select "Mark API" to expose its inputs</p>
    </div>
  {:else}
    <div
      class="mtb-inputs-list"
      use:dndzone={{ items: inputs, flipDurationMs, dragDisabled: locked }}
      onconsider={handleDndConsider}
      onfinalize={handleDndFinalize}
    >
      {#each inputs as item (item.id)}
        <div animate:flip={{ duration: flipDurationMs }}>
          <InspectorItem {item} />
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .mtb-tab-ui {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .mtb-tab-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.06));
  }

  .mtb-toolbar-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    background: transparent;
    border: 1px solid var(--border-color, rgba(255,255,255,0.1));
    border-radius: 6px;
    color: var(--descrip-text, rgba(255,255,255,0.5));
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .mtb-toolbar-btn:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.2);
  }

  .mtb-toolbar-btn.active {
    background: var(--mtb-api-color, #2930b0);
    border-color: var(--mtb-api-color, #2930b0);
    color: white;
  }

  .mtb-toolbar-label {
    font-size: 11px;
    color: var(--descrip-text, rgba(255,255,255,0.4));
  }

  .mtb-inputs-list {
    flex: 1;
    overflow-y: auto;
  }

  /* Empty state */
  .mtb-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    text-align: center;
  }

  .mtb-empty-icon {
    font-size: 32px;
    margin-bottom: 12px;
    opacity: 0.3;
  }

  .mtb-empty-text {
    font-size: 14px;
    font-weight: 500;
    color: var(--fg-color, #fafafa);
    margin: 0 0 8px;
  }

  .mtb-empty-hint {
    font-size: 12px;
    color: var(--descrip-text, rgba(255,255,255,0.4));
    margin: 0;
    max-width: 200px;
    line-height: 1.5;
  }
</style>
