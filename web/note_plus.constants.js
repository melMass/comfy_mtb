// web/note_plus.constants.js

export const DEFAULT_CSS = ''
export const DEFAULT_HTML = `<p style='color:red;font-family:monospace'>
    Note+
</p>`
export const DEFAULT_MD = '## Note+'
export const DEFAULT_MODE = 'markdown'
export const DEFAULT_THEME = 'one_dark'

export const DEMO_CONTENT = `
# @mtb/svelte-markdown.
## This is a subheader

[![embedded test](https://github.com/melMass/comfy_mtb/actions/workflows/test_embedded.yml/badge.svg)](https://github.com/melMass/comfy_mtb/actions/workflows/test_embedded.yml)
![home](https://repository-images.githubusercontent.com/649047066/a3eef9a7-20dd-4ef9-b839-884502d4e873)

<details>
<summary>More details about the inception of the project</summary>

\`\`\`js
class YesMan{
  constructor(){
    this.started = false
  }
}
\`\`\`
</details>

This is a paragraph. If it goes over the maximum width it will not automatically wrap unless it reaches the max-w of \`prose\` check [styles](/styles) for more info.

This component is useful for building some tools on top. Or even just a static system using svelte at its core. My personal blog is fully powered by **@mtb/svelte-markdown**

| And this is | A table |
|-------------|---------|
| With two    | columns |

We also support github callout:


> [!NOTE]
> Highlights information that users should take into account, even when skimming.
> [!TIP]
> Optional information to help a user be more successful.


> [!IMPORTANT]
> Crucial information necessary for users to succeed.

> [!WARNING]
> Critical content demanding immediate user attention due to potential risks.

> [!CAUTION]
> Negative potential consequences of an action.
`

export const THEMES = [
	'ambiance',
	'chaos',
	'chrome',
	'cloud9_day',
	'cloud9_night',
	'cloud9_night_low_color',
	'cloud_editor',
	'cloud_editor_dark',
	'clouds',
	'clouds_midnight',
	'cobalt',
	'crimson_editor',
	'dawn',
	'dracula',
	'dreamweaver',
	'eclipse',
	'github',
	'github_dark',
	'gob',
	'gruvbox',
	'gruvbox_dark_hard',
	'gruvbox_light_hard',
	'idle_fingers',
	'iplastic',
	'katzenmilch',
	'kr_theme',
	'kuroir',
	'merbivore',
	'merbivore_soft',
	'mono_industrial',
	'monokai',
	'nord_dark',
	'one_dark',
	'pastel_on_dark',
	'solarized_dark',
	'solarized_light',
	'sqlserver',
	'terminal',
	'textmate',
	'tomorrow',
	'tomorrow_night',
	'tomorrow_night_blue',
	'tomorrow_night_bright',
	'tomorrow_night_eighties',
	'twilight',
	'vibrant_ink',
	'vscode',
]

export const CSS_RESET = `
* {
  font-family: monospace;
  line-height: 1.25em;
}
.shiki{
    padding: 1em;
    width: 100%;
}
.markdown-callout-title {
    .octicon{
        fill:white;
    }
	/* background: var(--current-color); */
	color: var(--current-color);
	font-weight: bold;
	/* border-start-end-radius: var(--radius); */
	/* border-start-start-radius: var(--radius); */
	padding: 0.5em;
	padding-inline-start: 1em;
}
.markdown-callout-content {
	padding: 1em;
}
.markdown-callout {
	--radius: 8px;
	--current-color: purple;
	/* border-start-end-radius: var(--radius); */
	/* border-start-start-radius: var(--radius); */
	border-left: 3px solid var(--current-color);
	margin-bottom: 1em;
	margin-top: 1em;
}

.markdown-callout-tip {
	--text-color: whitesmoke;
	--current-color: #50e3c2;
}

.markdown-callout-note {
	--text-color: whitesmoke;
	--current-color: #0070f3;
}
.markdown-callout-important {
	--text-color: whitesmoke;
	--current-color: #7928ca;
}
.markdown-callout-warning {
	--current-color: #f5a623;
}
.markdown-callout-caution {
	--current-color: #e60000;
}


.note-plus-preview {
  display:flex;
  flex-direction:column;
  align-items: flex-start;
  width:95%;
  margin-left: 20px;
  margin-top:20px;
  /*background-color: rgba(255,0,0,0.5)!important;*/
}

/* allowed to be selected*/
h1, h2, h3, h4, h5, h6,a, p, ul, ol, dl, blockquote,details,summary  {
  pointer-events:auto;
  user-select:text;
}

h1, h2, h3, h4, h5, h6 {
  display:inline-block;
  margin: 0;
  padding: 0;
  font-weight: normal;
}

p, ul, ol, dl, blockquote {
  margin: 0.3em;
  padding: 0;
}

ul, ol {
  padding-left: 1em;
}

a {
  color: inherit;
  text-decoration: none;
  pointer-events: all;
  color: cyan;
}

img {
  padding: 1em 0;
  max-width: 100%;
}

iframe {
  max-width: 100%;
  height: auto;
  border:none;
  pointer-events:all;
}

blockquote {
  border-left: 4px solid #ccc;
  padding-left: 1em;
  margin-left: 0;
  font-style: italic;
}

pre, code {
  font-family: monospace;
}

table {
  border-collapse: collapse;
  width: 100%;
  border-bottom: 1px solid #000;
  margin: 1em 0;
}

th, td {
  border-left: 1px solid #000;
  border-right: 1px solid #000;
  padding: 8px;
  text-align: left;
}

th {
  border: 1px solid #000;
  background-color: rgba(0,0,0,0.5);
}

input[type="checkbox"] {
  margin-right: 10px;
}
`
