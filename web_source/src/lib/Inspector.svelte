<script lang="ts">
  import InspectorItem from './InspectorItem.svelte'
  import { slide, fade, fly, draw, scale } from 'svelte/transition'
  import { flip } from 'svelte/animate'
  import { dndzone } from 'svelte-dnd-action'
  import { dragMe, resizeHandle } from './actions.js'
  // import { draggable } from '@neodrag/svelte'

  import InspectorPreview from './InspectorPreview.svelte'
  import Tabs from './Tabs.svelte'
  import TabUi from './TabUI.svelte'
  import TabHelp from './TabHelp.svelte'
  import { inComfy } from './utils'
  import { onMount } from 'svelte'



  export let visible = true

  export let inputs = {
    nameA: {
      id: 1,
      name: 'positive',
      type: 'STRING',
    },
    nameB: {
      id: 2,
      name: 'architecture',
      type: 'COMBO',
      options: ['sd1.5', 'sd2', 'sd3', 'sdxl'],
    },
    nameC: {
      id: 3,
      name: 'seed',
      type: 'NUMBER',
    },
    nameD: {
      id: 4,
      name: 'background',
      type: 'COLOR',
    },
  }

  $: items = Object.keys(inputs).map((k) => {
    const input = inputs[k]
    console.log({ inputs })
    return {
      ...input,
      original_name: k,
    }
  })

  onMount(() => {
    console.log('Mounted API Inspector')
  })

  const flipDurationMs = 100
  let position = { x: 0, y: 0 }
  function handleDndConsider(e) {
    items = e.detail.items
  }
  function handleDndFinalize(e) {
    items = e.detail.items
  }
</script>

<!-- use:draggable={{ -->
<!--       defaultClassDragging: 'dragging', -->
<!--       defaultClassDragged: 'dragged', -->
<!--       handle: '.draggable', -->
<!--       legacyTranslate: false, -->
<!--       position, -->
<!--       onDrag: ({ offsetX, offsetY }) => { -->
<!--         pos = { x: offsetX, y: offsetY } -->
<!--       }, -->
<!--     }} -->
<!---->
{#if visible}
  <div transition:fade={{ duration: 60 }} use:resizeHandle class="container">
    <div id="inside">
      <div id="appbar" class="draggable" />
      <InspectorPreview />
      <Tabs
        items={[
          {
            label: 'Controls',
            value: 1,
            component: TabUi,
            props: {
              inputs: items,
            },
          },
          {
            label: 'Help',
            value: 2,
            component: TabHelp,
            props: {},
          },
        ]}
      />
      <div id="spacer"></div>
      <div id="main_buttons">
        <button
          on:click={() => {
            if (!inComfy()) {
              return
            }

            app.queuePrompt(0, 1)
          }}>Queue</button
        >
        <button>Cancel</button>
        <button>Export</button>
      </div>
      <div id="status">Idle</div>
      <!-- <section -->
      <!--   use:dndzone={{ items, flipDurationMs }} -->
      <!--   on:consider={handleDndConsider} -->
      <!--   on:finalize={handleDndFinalize} -->
      <!-- > -->
      <!--   {#each items as item, index (item.id)} -->
      <!--     <div animate:flip={{ duration: flipDurationMs }}> -->
      <!--       <InspectorItem {item} /> -->
      <!--     </div> -->
      <!--   {/each} -->
      <!-- </section> -->
    </div>
  </div>
{/if}

<style lang="scss">
  h2 {
    padding: 0;
  }
  #spacer {
    flex-grow: 1;
  }
  #notice {
    font-size: 12px;
    padding: 0 1em;
  }
  #appbar {
    height: 24px;
    width: 100%;
    background: #414141;
  }
  #status {
    user-select: none;
    width: 100%;
    background: rgba(0, 0, 0, 0.2);
    color: var(--descrip-text);
    font-size: 12px;
    text-align: center;
  }
  button {
    border-radius: 8px;
    border: 1px solid transparent;
    font-family: inherit;
    cursor: pointer;
    transition: border-color 0.25s;
  }
  /*button:hover {
    border-color: #646cff;
  }*/
  #main_buttons {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  #inside {
    height: 100%;
    overflow: auto;
    display: flex;
    flex-direction: column;
    h2 {
      text-align: center;
    }
  }
  .container {
    /* top: 0; */
    /* left: 0; */
    border-radius: 12px;
    background-color: var(--bg-color);
    min-height: 160px;
    /* overflow: hidden; */

    font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    font-weight: 400;

    color-scheme: light dark;
    color: rgba(255, 255, 255, 0.87);
    /* background-color: #242424; */

    font-synthesis: none;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;

    /* position: absolute; */
  }
</style>
