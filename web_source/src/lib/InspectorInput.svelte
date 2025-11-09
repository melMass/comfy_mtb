<script lang="ts">
  // import { Palette } from '@untemps/svelte-palette'
  import ColorPicker, { ChromeVariant } from 'svelte-awesome-color-picker'

  // import { portal } from 'svelte-portal'
  import Wrapper from './Wrapper.svelte'


  let element:HTMLElement

  let {
    item = { type: "NUMBER" },
    onInput = (e) => console.log(e),
    id = undefined,
    value = $bindable("")
  } = $props()

  const shared = {
    autocomplete: 'off',
    'data-lpignore': true,
    'data-form-type': 'other',
  }
  const colors = ['#865C54', '#8F5447', '#A65846', '#A9715E', '#AD8C72']
</script>

{#if item.type == 'NUMBER'}
  <input
    oninput={onInput}
    {...shared}
    {id}
    bind:value
    bind:this={element}
    type="number"
  />
{:else if item.type == 'MODEL' || item.type == 'COMBO'}
  {@const options = item.options || item.widgets?.[0]?.options?.values || []}
  <select oninput={onInput} bind:value {...shared} {id} bind:this={element}>
    {#each options as option}
      <option selected={item.widget?.value === option} value={option}
        >{option}</option
      >
    {/each}
  </select>
{:else if item.type == 'STRING'}
  {#if item.widgets?.[0]?.type === 'customtext'}
    <textarea
      oninput={onInput}
      bind:value
      {...shared}
      {id}
      bind:this={element}
    ></textarea>
  {:else}
    <input
      oninput={onInput}
      bind:value
      {...shared}
      {id}
      type="text"
      bind:this={element}
    />
  {/if}
{:else if item.type === 'COLOR'}
  <ColorPicker
    label=""
    bind:hex={value}
    components={{ ...ChromeVariant, wrapper: Wrapper }}
    sliderDirection="horizontal"
    --picker-z-index="9000"
    oninput={(event) => {
      value = event.detail.hex
    }}
  />
{:else}
  <input oninput={onInput} bind:value {...shared} {id} bind:this={element} />
{/if}

<style>
  #test {
    overflow: visible;
  }
  input,
  select {
    border-radius: 6px;
    width: 100%;
  }
</style>
