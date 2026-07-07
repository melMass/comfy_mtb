<script lang="ts">
  import type { Component as SvelteComponent } from 'svelte'

  interface TabItem {
    label: string
    value: number
    component: SvelteComponent<unknown>
    props: Record<string, unknown>
  }

  let { items = [] as TabItem[] } = $props()

  let activeTabValue = $state(items[0]?.value ?? 1)

  function handleClick(value: number) {
    return () => {
      activeTabValue = value
    }
  }
</script>

<div class="mtb-tabs">
  <nav class="mtb-tabs-nav" role="tablist">
    {#each items as item}
      <button
        class="mtb-tab"
        class:active={activeTabValue === item.value}
        role="tab"
        aria-selected={activeTabValue === item.value}
        onclick={handleClick(item.value)}
      >
        {item.label}
      </button>
    {/each}
  </nav>

  <div class="mtb-tabs-content">
    {#each items as item}
      {#if activeTabValue === item.value}
        <svelte:component this={item.component} {...item.props} />
      {/if}
    {/each}
  </div>
</div>

<style>
  .mtb-tabs {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .mtb-tabs-nav {
    display: flex;
    gap: 4px;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.06));
  }

  .mtb-tab {
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--descrip-text, rgba(255,255,255,0.5));
    background: transparent;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .mtb-tab:hover {
    color: var(--fg-color, #fafafa);
    background: rgba(255,255,255,0.05);
  }

  .mtb-tab.active {
    color: var(--fg-color, #fafafa);
    background: rgba(255,255,255,0.1);
  }

  .mtb-tabs-content {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
</style>
