<script lang="ts">
  import ColorPicker, { ChromeVariant } from 'svelte-awesome-color-picker'
  import Wrapper from './Wrapper.svelte'

  let element: HTMLElement

  let {
    item = { type: 'NUMBER' },
    onInput = (e: Event) => console.log(e),
    id = undefined,
    value = $bindable(''),
  } = $props()

  const shared = {
    autocomplete: 'off',
    'data-lpignore': true,
    'data-form-type': 'other',
  }
</script>

<div class="mtb-input-wrapper">
  {#if item.type === 'NUMBER'}
    <input
      class="mtb-input"
      oninput={onInput}
      {...shared}
      {id}
      bind:value
      bind:this={element}
      type="number"
    />
  {:else if item.type === 'MODEL' || item.type === 'COMBO'}
    {@const options = item.options || item.widgets?.[0]?.options?.values || []}
    <select class="mtb-select" oninput={onInput} bind:value {...shared} {id} bind:this={element}>
      {#each options as option}
        <option selected={item.widget?.value === option} value={option}>{option}</option>
      {/each}
    </select>
  {:else if item.type === 'STRING'}
    {#if item.widgets?.[0]?.type === 'customtext'}
      <textarea
        class="mtb-textarea"
        oninput={onInput}
        bind:value
        {...shared}
        {id}
        bind:this={element}
        rows="3"
      ></textarea>
    {:else}
      <input
        class="mtb-input"
        oninput={onInput}
        bind:value
        {...shared}
        {id}
        type="text"
        bind:this={element}
      />
    {/if}
  {:else if item.type === 'BOOLEAN'}
    <label class="mtb-toggle">
      <input
        type="checkbox"
        checked={value === true || value === 'true'}
        onchange={(e) => {
          value = e.currentTarget.checked
          onInput(e)
        }}
      />
      <span class="mtb-toggle-slider"></span>
      <span class="mtb-toggle-label">{value ? 'On' : 'Off'}</span>
    </label>
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
    <input class="mtb-input" oninput={onInput} bind:value {...shared} {id} bind:this={element} />
  {/if}
</div>

<style>
  .mtb-input-wrapper {
    width: 100%;
  }

  /* Base input styles */
  .mtb-input,
  .mtb-select,
  .mtb-textarea {
    width: 100%;
    padding: 8px 12px;
    font-size: 13px;
    font-family: inherit;
    color: var(--fg-color, #fafafa);
    background: var(--comfy-input-bg, rgba(255,255,255,0.05));
    border: 1px solid var(--border-color, rgba(255,255,255,0.1));
    border-radius: 6px;
    outline: none;
    transition: all 0.15s ease;
  }

  .mtb-input:hover,
  .mtb-select:hover,
  .mtb-textarea:hover {
    border-color: var(--border-color, rgba(255,255,255,0.2));
  }

  .mtb-input:focus,
  .mtb-select:focus,
  .mtb-textarea:focus {
    border-color: var(--mtb-api-color, #2930b0);
    box-shadow: 0 0 0 3px rgba(41, 48, 176, 0.15);
  }

  /* Number input */
  .mtb-input[type='number'] {
    font-variant-numeric: tabular-nums;
  }

  .mtb-input[type='number']::-webkit-inner-spin-button,
  .mtb-input[type='number']::-webkit-outer-spin-button {
    opacity: 0;
  }

  .mtb-input[type='number']:hover::-webkit-inner-spin-button,
  .mtb-input[type='number']:hover::-webkit-outer-spin-button {
    opacity: 1;
  }

  /* Select */
  .mtb-select {
    cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23888' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 12px center;
    padding-right: 36px;
  }

  .mtb-select option {
    background: var(--comfy-menu-bg, #1a1a1a);
    color: var(--fg-color, #fafafa);
  }

  /* Textarea */
  .mtb-textarea {
    resize: vertical;
    min-height: 60px;
    line-height: 1.5;
  }

  /* Toggle switch */
  .mtb-toggle {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
  }

  .mtb-toggle input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
  }

  .mtb-toggle-slider {
    position: relative;
    width: 36px;
    height: 20px;
    background: var(--border-color, rgba(255,255,255,0.15));
    border-radius: 10px;
    transition: all 0.2s ease;
  }

  .mtb-toggle-slider::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s ease;
  }

  .mtb-toggle input:checked + .mtb-toggle-slider {
    background: var(--mtb-api-color, #2930b0);
  }

  .mtb-toggle input:checked + .mtb-toggle-slider::after {
    transform: translateX(16px);
  }

  .mtb-toggle-label {
    font-size: 12px;
    color: var(--descrip-text, rgba(255,255,255,0.6));
  }
</style>
