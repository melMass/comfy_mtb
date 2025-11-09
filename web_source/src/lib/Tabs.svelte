<script lang="ts">
  
  let {
    items = [],
    activeTabValue = 1,
  } = $props()

  const handleClick = (tabValue: number) => () => {activeTabValue = tabValue}
</script>

<ul>
  {#each items as item}
    <li class={activeTabValue === item.value ? 'active' : ''}>
      <span  onclick={handleClick(item.value)}>{item.label}</span>
    </li>
  {/each}
</ul>
{#each items as item}
  {#if activeTabValue == item.value}
    {@const Component = item.component}
    <div class="box">
      <Component {...item.props} />
      
    </div>
  {/if}
{/each}

<style>
  .box {
    margin-bottom: 10px;
    padding: 0.5em;
    border: 1px solid var(--border-color);
    border-radius: 0 0 0.5rem 0.5rem;
    border-top: 0;
  }
  ul {
    display: flex;
    flex-wrap: wrap;
    padding-left: 0;
    margin-bottom: 0;
    list-style: none;
    border-bottom: 1px solid var(--border-color);
  }
  li {
    margin-bottom: -1px;
  }

  span {
    border: 1px solid transparent;
    border-top-left-radius: 0.25rem;
    border-top-right-radius: 0.25rem;
    display: block;
    padding: 0.5rem 1rem;
    cursor: pointer;
  }

  span:hover {
    border-color: #e9ecef #e9ecef #dee2e6;
  }

  li.active > span {
    color: var(--fg-color);
    background-color: var(--comfy-menu-bg);
    border-color: var(--border-color);
  }
</style>
