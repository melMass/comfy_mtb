<script lang="ts">
  import { portal } from 'svelte-portal'

  export let wrapper
  export let isOpen
  export let isPopup
  /* svelte-ignore unused-export-let */
  export let toRight
</script>

<div
  use:portal={'#inside'}
  class="wrapper"
  bind:this={wrapper}
  class:isOpen
  class:isPopup
  role={isPopup ? 'dialog' : undefined}
  aria-label="color picker"
>
  <slot />
</div>

<style>
  div {
    padding: 8px 5px 5px 8px;
    margin: 0 10px 10px;
    border: 1px solid black;
    border-radius: 12px;
    display: none;
    width: max-content;
  }
  .isOpen {
    display: block;
  }
  .isPopup {
    position: absolute;
    top: 30px;
    z-index: var(--picker-z-index, 2);
  }
</style>
