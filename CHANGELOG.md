# Changelog

This is an automated changelog based on the commits in this repository.

Check the notes in the [releases](https://github.com/melMass/comfy_mtb/releases) for more information.
## [main] - 2025-04-16

### Bug Fixes

- 🐛 note+ breaking wfs ([af42134](https://github.com/melMass/comfy_mtb/commit/af421340286b234e4c0cfcd4143a9d8726ebf3d1))
- 🐛 ColorCorrect clamp issue ([a85e57b](https://github.com/melMass/comfy_mtb/commit/a85e57b18c7d3c765131873ffff523244ca9be73))
- 🐛 Whisper chunks processing ([8bf3545](https://github.com/melMass/comfy_mtb/commit/8bf3545fec5b2a180607d40394b025a1e09c14b6))
- 🐛 stackImages move to device ([d87e52e](https://github.com/melMass/comfy_mtb/commit/d87e52ea2c112fd95f257dcd6a54a5db77a34fc3))
- 🐛 bbox upscale from center ([55261bd](https://github.com/melMass/comfy_mtb/commit/55261bda7c33d088b62c5483e4483201e5a9ce77))
- 🐛 add MASK support for PickFromBatch ([0d264b9](https://github.com/melMass/comfy_mtb/commit/0d264b90a78d5a6719fb3ce71f4e9a642db4c950))
- 🐛 use addDOMWidget for Debug node ([46af602](https://github.com/melMass/comfy_mtb/commit/46af6027d6c87d0c29b8bb0fd1cc1dbdae993629))
- 🐛 use "modern" notation in toDevice ([b7ca8ed](https://github.com/melMass/comfy_mtb/commit/b7ca8ed1c6e117b71afd7696f55dcc3dbd5bad08))
- 🐛 handle missing submodules ([d61da30](https://github.com/melMass/comfy_mtb/commit/d61da304099ff5e4528e4beb1ecc2eb83cabaaa1))
- 🐛 add warnings about what each IO mode can do ([6608c0b](https://github.com/melMass/comfy_mtb/commit/6608c0b6d1cf8f7a9901214096f8c78bfe17056f))
- 🐛 soft deprecate compression h264 ([a757e1c](https://github.com/melMass/comfy_mtb/commit/a757e1c98b2abbd2221a15b77e89d772e02d1d82))
- 🐛 limit packages allowed to be installed from API ([d6e004c](https://github.com/melMass/comfy_mtb/commit/d6e004cce2c32f8e48b868e66b89f82da4887dc3))
- 🐛 ensure default settings (io sidebar) ([ed17fa2](https://github.com/melMass/comfy_mtb/commit/ed17fa2ef4688aadf305a6d51b32c13a0efd22d6))
- 🐛 spawn colour picker at pointer location ([e5482ae](https://github.com/melMass/comfy_mtb/commit/e5482aee5e3de07e8f055b3edc0fccc0e0f75c14)) by [@webfiltered](https://github.com/webfiltered) in [#223](https://github.com/melMass/comfy_mtb/pull/223)
- 🐛 i/o sidebar for custom paths ([62469a4](https://github.com/melMass/comfy_mtb/commit/62469a4dd96e32509171aad74fcae8d2bb0ec593))

### Features

- ⚡ add BatchFromFolder ([9618513](https://github.com/melMass/comfy_mtb/commit/96185132b83c182032e9f6e822561eb5699af517))
- ⚡ add use_normalized to TransformBatch2D ([d4a31bf](https://github.com/melMass/comfy_mtb/commit/d4a31bf19c2863df8dfc4cb9a3cd6683304949e4))
- [**breaking**] ⚡ add support for masks in BatchFLoatMath ([fc7ba08](https://github.com/melMass/comfy_mtb/commit/fc7ba084f6ed7880e88e28eb448ab0bd7d796824))
- ✨ add use_normalized to TransformImage ([4516aa9](https://github.com/melMass/comfy_mtb/commit/4516aa9cb4fcb12c946999d6dcc1501cc09011a3))
- ✨ add regex support for String Replace ([78946b0](https://github.com/melMass/comfy_mtb/commit/78946b0fa3c3cf5dfcee8c7c4c0921b722d09d1e)) by [@poetryiii](https://github.com/poetryiii) in [#233](https://github.com/melMass/comfy_mtb/pull/233)
- ✨ update diarization to 3.1 ([c30408f](https://github.com/melMass/comfy_mtb/commit/c30408f96d4df9c7d35545654401162090a74305)) by [@numz](https://github.com/numz)
- ✨ add "workflow" query to /mtb/view endpoint ([eb7cf89](https://github.com/melMass/comfy_mtb/commit/eb7cf89f173b2342b04e7b61dca3d12cfaf65bdb))
- ✨ add stretch_x and stretch_y to TransformImage ([22fce6f](https://github.com/melMass/comfy_mtb/commit/22fce6fdda135cbb1f1aad42c86aae166cba81b5))
- ✨ add AudioDuration node ([f471497](https://github.com/melMass/comfy_mtb/commit/f47149746ac1e418cda2007c38aafbb03946ce22))
- ✨ basic whisper nodes ([83cfc5c](https://github.com/melMass/comfy_mtb/commit/83cfc5c723d1a572af67ad14b52be4f8371a3c5f))
- ✨ add BboxForDimensions ([9405784](https://github.com/melMass/comfy_mtb/commit/940578476438eaa6a42e0056f1b7b319ee585334))
- ✨ improve the debug node ([cf7a9c4](https://github.com/melMass/comfy_mtb/commit/cf7a9c41e81e8dd461ab9dfa3c05bb8e2cdf2a67))
- ✨ add BatchImageToSublist and counterpart ([00173fa](https://github.com/melMass/comfy_mtb/commit/00173fa3fbca4c5b1ff3016cc5139705ce61ec20))
- ✨ add TensorOps ([a8cf465](https://github.com/melMass/comfy_mtb/commit/a8cf4650ff5cbd4975ef954b5829c772ee53250c))
- ✨ live update outputs grid ([7f7a62f](https://github.com/melMass/comfy_mtb/commit/7f7a62f832c865a13b9181daee79d3cfc21581e2)) by [@christian-byrne](https://github.com/christian-byrne) in [#229](https://github.com/melMass/comfy_mtb/pull/229)
- ✨ add SaveImage passthrough ([0eeb707](https://github.com/melMass/comfy_mtb/commit/0eeb707f34f51142def8e0ef7d351ee5028cb5e0))
- ✨ add filtering to TransformImage ([bae26a0](https://github.com/melMass/comfy_mtb/commit/bae26a07fb02dd518c621eba28986a51c5d086bc))
- ✨ add support for video in I/O sidebar ([c92d99a](https://github.com/melMass/comfy_mtb/commit/c92d99a8a37a64cfc285296f21452c4927a22774))
- ✨ add an extra static input to Stack Images ([3f6d082](https://github.com/melMass/comfy_mtb/commit/3f6d08294096918d50101a19083f9134305cc8c9)) in [#222](https://github.com/melMass/comfy_mtb/pull/222)
- ✨ add support for subdirs (i/o sidebar) ([52bd76e](https://github.com/melMass/comfy_mtb/commit/52bd76e19c8bd7e72986900e5dbfade0457ef7e0))
- ✨ add Batch Sequence Nodes ([827c64c](https://github.com/melMass/comfy_mtb/commit/827c64c43d52ebfb8acd2e5c4491c4b66e6b8f40))
- ✨ add support for more formats (I/O sidebar) ([8c629be](https://github.com/melMass/comfy_mtb/commit/8c629bee186b5ac991058018a788e4a836eef630))

### Miscellaneous Tasks

- 🧹 bump version ([d093d76](https://github.com/melMass/comfy_mtb/commit/d093d76efd87474a3ca82858147255038060ab17))
- 🧹 small adjustments ([01107c4](https://github.com/melMass/comfy_mtb/commit/01107c45f8539ff7c579e08e2a9075d93781b9a2))
- 🤖 update publish action workflow with permissions and version constraints ([0e48aaa](https://github.com/melMass/comfy_mtb/commit/0e48aaa3e4f1e440a5d7ab42df56b728ced03aca)) by [@robinjhuang](https://github.com/robinjhuang) in [#237](https://github.com/melMass/comfy_mtb/pull/237)
- 🧹 basic standalone detection ([3a25526](https://github.com/melMass/comfy_mtb/commit/3a25526e818a1af8f886d2ad5c27101c4a0caa8b))
- 🧹 rename type ([edcb3da](https://github.com/melMass/comfy_mtb/commit/edcb3da08bff66f9adcef8dcd37c3925e64d0135))
- 🧹 update env file ([fc908ba](https://github.com/melMass/comfy_mtb/commit/fc908ba0a528523b7c1e37e34fb32f430746de0d))
- 🧹 dev ([9a94371](https://github.com/melMass/comfy_mtb/commit/9a943714aada107bfd236e00fa1063872db7a834))
- 🧹 apply formatting ([58ae89f](https://github.com/melMass/comfy_mtb/commit/58ae89f8e0f0f8b42825722a6aebc04da39847b1))

### Refactor

- 📦 add model autodownload ([147edcf](https://github.com/melMass/comfy_mtb/commit/147edcfcbc09dd27a0c787f9da568fb850c3308a))

### Wip

- 🚧 loop drawing ([ead4b34](https://github.com/melMass/comfy_mtb/commit/ead4b34e6dd03ea4ed309b246ef31c995325aa08))

## New Contributors
* [@poetryiii](https://github.com/poetryiii) made their first contribution in [#233](https://github.com/melMass/comfy_mtb/pull/233)
* [@numz](https://github.com/numz) made their first contribution in [#](https://github.com/melMass/comfy_mtb/pull/)
* [@webfiltered](https://github.com/webfiltered) made their first contribution in [#223](https://github.com/melMass/comfy_mtb/pull/223)
## [0.2.0] - 2024-12-08

### Bug Fixes

- 🐛 remove mtb sidebar ([b0d52f7](https://github.com/melMass/comfy_mtb/commit/b0d52f73051368df6de2d1e10ad28ca56df72803))
- 🐛 always enable the I/O sidebar ([ec1cb1a](https://github.com/melMass/comfy_mtb/commit/ec1cb1ac17d14670aa756dfb1ae7542397b12559))
- 🐛 ui shifts on animation builder ([ecbb220](https://github.com/melMass/comfy_mtb/commit/ecbb220de6a05f2e506ec43f2b786be983166157))
- 🐛 category for settings ([b6fa571](https://github.com/melMass/comfy_mtb/commit/b6fa571fd2096ace60d03cab42dba9ca37d0cb27)) in [#211](https://github.com/melMass/comfy_mtb/pull/211)
- 🐛 new UI issues ([f272526](https://github.com/melMass/comfy_mtb/commit/f272526bfc5da95e95d42cb4c613a0b9585b2577))
- 🐛 disable old BOOL widget (legacy) ([8596b81](https://github.com/melMass/comfy_mtb/commit/8596b8184edb484c907475a77ac1dc9e4a5c92af))
- 🐛 pass ONNX providers explicitely ([43092e4](https://github.com/melMass/comfy_mtb/commit/43092e44a4ea17f90fcfb12372da634fe4b79557))
- 🐛 typo in mtb_widgets error catch ([80b5a0c](https://github.com/melMass/comfy_mtb/commit/80b5a0ca7459763e7662421bccd8636976eefddd)) by [@christian-byrne](https://github.com/christian-byrne) in [#197](https://github.com/melMass/comfy_mtb/pull/197)
- 🐛 doc widget sidebar offset in the new ui ([81b3bc1](https://github.com/melMass/comfy_mtb/commit/81b3bc1651f06ad2fa7938f810d3f406f5e7c41c))
- 🐛 don't fallback to eval ([997d2fb](https://github.com/melMass/comfy_mtb/commit/997d2fb13af6aadf36873ea2ea3317e56f405aef))
- 🐛 rework main utils ([c99b081](https://github.com/melMass/comfy_mtb/commit/c99b0812ab4a4183ef9298fb8a7c954bc7c858b2))
- 🐛 MaskToImage ([821a0ef](https://github.com/melMass/comfy_mtb/commit/821a0ef42735a0a97ab82be22a4fdc67c9cfc80e))

### Documentation

- 📚 update wiki ([e17c6e2](https://github.com/melMass/comfy_mtb/commit/e17c6e29f5111bf5085b1fe6f764cfd1aae709f2))
- 📚 remove link ([5bc125d](https://github.com/melMass/comfy_mtb/commit/5bc125d2f08470c8900dfd89deca721835848917))
- 📚 clean readme ([333f646](https://github.com/melMass/comfy_mtb/commit/333f646ab1959d2c944fb046275cc93a545d557c))

### Features

- ✨ add h264 compression node ([e32d1e0](https://github.com/melMass/comfy_mtb/commit/e32d1e02df5e3a9351f829513f7ee3ffb2934be4))
- ✨ add postshot nodes ([27e03fa](https://github.com/melMass/comfy_mtb/commit/27e03fa23efffda461c6975b15fe3964de476cb3))
- ✨ improve the I/O sidebar ([cd9e614](https://github.com/melMass/comfy_mtb/commit/cd9e614b1a385d6b06eacfaad62def1d69f09808)) in [#193](https://github.com/melMass/comfy_mtb/pull/193)
- ✨ add UpscaleBBoxBy ([74af5c6](https://github.com/melMass/comfy_mtb/commit/74af5c6499ef5dd73ce66c4c21b8c3507d69b037))
- ✨ simplified sidebar and backend ([22f7c30](https://github.com/melMass/comfy_mtb/commit/22f7c3037345a866c9ff0b06f6689748021cee63))
- ✨ add Interpolate Condition ([0133fb9](https://github.com/melMass/comfy_mtb/commit/0133fb93bc944d0dd7593b89b36e5b2676d9397a))
- ✨ dump of wip things... ([cf7d305](https://github.com/melMass/comfy_mtb/commit/cf7d30507e7e449c4489e6a1ca159d3d0486bc55))
- ✨ use the new parser for documentations ([4e593bb](https://github.com/melMass/comfy_mtb/commit/4e593bb30be561e39f1790e3514f60bb39e5a261))
- ✨ add @mtb/markdown-parser bundles ([097ca33](https://github.com/melMass/comfy_mtb/commit/097ca33b8e7b27148e183e91712dc34d98d1a69b))
- ✨ add VitMatte nodes ([896a025](https://github.com/melMass/comfy_mtb/commit/896a025006f9c7809c5e0776393a28f908be8950))
- ✨ add ColorCorrectGPU ([9651a70](https://github.com/melMass/comfy_mtb/commit/9651a7034120589b059329b21688708e42772453))
- ✨ add Swap FG/BG colors to MaskToImage ([57683c3](https://github.com/melMass/comfy_mtb/commit/57683c3c7d299a117a26526d52de4c26f2ec0f69))
- ✨ add Extract coordinates ([f99f92e](https://github.com/melMass/comfy_mtb/commit/f99f92e8f7b2d6fac56f7f40049715910e15cfee))
- ✨ add AudioCut ([5681b46](https://github.com/melMass/comfy_mtb/commit/5681b464adce395086712b61159b2694150b8027))
- ✨ add AudioStack ([8d0fcee](https://github.com/melMass/comfy_mtb/commit/8d0fcee2f3decc1cbbf3b850332e6b2a022e1377))
- ✨ add AudioSequence node ([1078fc6](https://github.com/melMass/comfy_mtb/commit/1078fc6f0fb225b52536f25ec6a9fa0456a90595))
- ✨ add Split Bbox node ([9007a70](https://github.com/melMass/comfy_mtb/commit/9007a70aa0d6b2ead0f68f7aff8ae8e3c4f3624f))
- ✨ update lerp example ([1a0ebd5](https://github.com/melMass/comfy_mtb/commit/1a0ebd5173687784f279a9c2184c89fb3be01dc5))

### Miscellaneous Tasks

- 🧹 bump minor ([50cb6f5](https://github.com/melMass/comfy_mtb/commit/50cb6f5ed6e5d9fecb9733ef3f7852b8500005e9))
- 🧹 add worktree to gitignores ([9ccf572](https://github.com/melMass/comfy_mtb/commit/9ccf572a158caeab9bff53853e8f6fb85b76776d))
- 🧹 remove dupe code ([e099d58](https://github.com/melMass/comfy_mtb/commit/e099d581a7627c3a66d2e3e6df3a701b0e5f31b7))
- 🧹 update externs ([784fb01](https://github.com/melMass/comfy_mtb/commit/784fb0145b7421e2730b52237ce6a8b63b189191))
- 🧹 add pathlibed inputs to utils ([a825504](https://github.com/melMass/comfy_mtb/commit/a825504bdd67e3461be8118119e0becc35f8af40))
- 🧹 disable Constant ([22190cd](https://github.com/melMass/comfy_mtb/commit/22190cd25ee590595f8f19e75a9a6c539699622b))
- 🧹 new ui is default, flag for old ui ([a976adb](https://github.com/melMass/comfy_mtb/commit/a976adbb39a13b4cd76f224ebba40c604900c862))
- 🧹 add methods to shared ([f8829fc](https://github.com/melMass/comfy_mtb/commit/f8829fcb373e0f9bc4f0ad36c939f372349943bf))
- 🧹 add an old_ui flag to my launcher ([dbdf276](https://github.com/melMass/comfy_mtb/commit/dbdf27664cd207dbbc69b8d635adcd59ed8d269a))
- 🧹 move qrcode to his own file ([7d5569e](https://github.com/melMass/comfy_mtb/commit/7d5569e5c1e0f0b6ccb505a02f74640139d6aaf9))

## [0.1.6] - 2024-07-03

### Bug Fixes

- 🐛 menu callback issue ([d64fac4](https://github.com/melMass/comfy_mtb/commit/d64fac4b74e0590acde5e3b8edd4a2f715448cf5))

### Documentation

- 📚 Update requirements file in INSTALL.md ([f25f6bd](https://github.com/melMass/comfy_mtb/commit/f25f6bdcd13d50f9d383065321320b0ce6a03214)) by [@elthariel](https://github.com/elthariel) in [#186](https://github.com/melMass/comfy_mtb/pull/186)

### Features

- ✨ add alpha channel support for faceswap/restore ([d6343e1](https://github.com/melMass/comfy_mtb/commit/d6343e1860f46947e93758f8bba03857c9326b38))

### Miscellaneous Tasks

- 🧹 better classname extraction ([d687497](https://github.com/melMass/comfy_mtb/commit/d687497d8041ab5d77bd31909592def6e4d0e7f6))
- 🤖 limit release only to tags ([4eebdd8](https://github.com/melMass/comfy_mtb/commit/4eebdd8b8bff73c3db4f0248da8dac7d67cb310b))
- 🧹 runner ([fb34671](https://github.com/melMass/comfy_mtb/commit/fb34671ee6fe80b965fe576c279ed1ff77a358f2))
- 🤖 only publish on tag ([f1b4846](https://github.com/melMass/comfy_mtb/commit/f1b484617a917d38d9b3658d8920aa7dec672a79))
- 🧹 small fixes ([4507842](https://github.com/melMass/comfy_mtb/commit/4507842a706141977a6a68945c36e977c358d91a))

## New Contributors
* [@elthariel](https://github.com/elthariel) made their first contribution in [#186](https://github.com/melMass/comfy_mtb/pull/186)
## [0.1.5] - 2024-06-21

### Bug Fixes

- 🐛 keep the last model match instead of first ([1edc2cd](https://github.com/melMass/comfy_mtb/commit/1edc2cd10de81297e7a895009d358813e79b70ba))
- 🐛 properly initialize the curve value ([35622e3](https://github.com/melMass/comfy_mtb/commit/35622e3a5e58103a8f5b150556b85e97e31555e1))
- 🐛 ImageCompare improvements ([acc2d68](https://github.com/melMass/comfy_mtb/commit/acc2d687d596bf82c2075f9a24003eacf18adfe7)) by [@christian-byrne](https://github.com/christian-byrne) in [#176](https://github.com/melMass/comfy_mtb/pull/176)
- 🐛 repetitive warning ([780c52f](https://github.com/melMass/comfy_mtb/commit/780c52f03aca3079a1b695510341486720004bec)) by [@vxkj1211](https://github.com/vxkj1211) in [#177](https://github.com/melMass/comfy_mtb/pull/177)
- 🐛 add back was conversion node ([349a852](https://github.com/melMass/comfy_mtb/commit/349a8524c6f7fcab4a124cacb60bfbef1463cf1b))
- 🐛 drag lag on documentation resize handle ([15330ea](https://github.com/melMass/comfy_mtb/commit/15330eab655f66214d3c25fd237679f090175c32))
- 🐛 kwarg typo ([1571782](https://github.com/melMass/comfy_mtb/commit/1571782d012b83bce32a065e700f9a587db234d2))
- 🐛 seed of PlotBatchFloat ([5b40302](https://github.com/melMass/comfy_mtb/commit/5b4030288d43c79859c9706a12aa0f8b7dea190f))
- 🐛 forceInput for FLOAT <-> FLOATS converters ([5a0ef0d](https://github.com/melMass/comfy_mtb/commit/5a0ef0dadd01fd5937ed0715d829d6a456f96318))
- 🐛 FLOAT always need options to be set ([967e72f](https://github.com/melMass/comfy_mtb/commit/967e72fc66780685f8192cb8fe13ba66b9326f63))
- 🐛 remove doc if opened on node delete ([bee3f47](https://github.com/melMass/comfy_mtb/commit/bee3f47a14ddb92b3760098666bf75dc7d37f1e4))
- 🐛 for documentation on HiDPI ([b11346a](https://github.com/melMass/comfy_mtb/commit/b11346aba88d9f1dac3b6b42c691979cc0978b6f))
- 🐛 never remove input 0 of dynamic inputs ([30982fa](https://github.com/melMass/comfy_mtb/commit/30982fa48829c3fc2a6745ce5a07537a3d94b2f9))
- 🐛 use the same fix as dynamicInputs for debug ([92b7990](https://github.com/melMass/comfy_mtb/commit/92b79906cd2ee1b4ca3ff25378d7786b5a47cb75))
- 🐛 missing numberInput ([76f365b](https://github.com/melMass/comfy_mtb/commit/76f365b5eee165c76f3da7d2e3950786685bc08b))
- 🐛 better curve ([da67e76](https://github.com/melMass/comfy_mtb/commit/da67e766c2f700dd9e2f51a5bafe07c612904f5d))
- 🐛 prepend MTB_ to all classes ([b1d74ad](https://github.com/melMass/comfy_mtb/commit/b1d74adb15166e3e5eb9cf92d6148e4644bed346))
- 🐛 dynamic connections ([652ac3f](https://github.com/melMass/comfy_mtb/commit/652ac3f3b971582b02115177fd6f7a9d3d7295df))
- 🐛 remaining issue before merge ([100067a](https://github.com/melMass/comfy_mtb/commit/100067a645194366426f29b085bf25d0623f4fac))
- 🐛 debug issues ([7807449](https://github.com/melMass/comfy_mtb/commit/7807449e6dcc01cfdb7f0eb818569184c8b41af2))
- 🐛 errors when insightface's folder missing ([e838c04](https://github.com/melMass/comfy_mtb/commit/e838c04758402250fd3464d6cd6a6f872e8cef29))
- 🐛 typo ([e40ad7a](https://github.com/melMass/comfy_mtb/commit/e40ad7a574f961ebe1f338b97214da5cbadcc529))
- 🐛 better defaults (cont) ([1da483a](https://github.com/melMass/comfy_mtb/commit/1da483a8baa6a893f1adb05ef79b90c4412c3834))
- 🐛 better defaults for Autopan ([5eff38b](https://github.com/melMass/comfy_mtb/commit/5eff38b387d22206d39c08e435806f9d03992feb))
- 🐛 dynamic inputs ([9ab20a0](https://github.com/melMass/comfy_mtb/commit/9ab20a0ab50b1656ded9a84c13769fd2d547f2d2))
- 🐛 bundle ace editor ([7c35582](https://github.com/melMass/comfy_mtb/commit/7c3558273bebc0754c802720e705232f220a0da4))
- 🐛 image to mask ([f16d576](https://github.com/melMass/comfy_mtb/commit/f16d576f6f0e83fc2fafd2d1f29b2edeb00d3197))
- 🐛 prepend MTB to classnames ([e56508c](https://github.com/melMass/comfy_mtb/commit/e56508c2078155f053e7f11d538a048df6a5b18b))
- 🐛 allow smaller values in BatchTransform ([9a4b27d](https://github.com/melMass/comfy_mtb/commit/9a4b27d2e05e8ebe31f58a21db94bd3a54ed23d9))
- 🐛 add category for virtual note+ ([eeac8c0](https://github.com/melMass/comfy_mtb/commit/eeac8c002ad1f9e461418fb66b9338e969259e58))
- 🐛 make image feed of by default ([df0a98b](https://github.com/melMass/comfy_mtb/commit/df0a98b94a4a9388811bc8786e820ec892919c1a))
- 🐛 support batch masks (colored image node) ([2465ffb](https://github.com/melMass/comfy_mtb/commit/2465ffb0d3b052fb78559394dbb550bba59b97a3))
- 🐛 support pillow < 10 ([48f91b7](https://github.com/melMass/comfy_mtb/commit/48f91b74e2c7ef6d31c094eafa5332784a275a8b))
- 🐛 image rotation bug ([54ff658](https://github.com/melMass/comfy_mtb/commit/54ff6583ded0ed4054f8e5d7fadf0b2350259dce)) by [@hongminpark](https://github.com/hongminpark) in [#154](https://github.com/melMass/comfy_mtb/pull/154)
- 🐛 font fallback ([9fccdee](https://github.com/melMass/comfy_mtb/commit/9fccdee82d721e88c64d2292c209fec869524dd2))
- ✨ optional inputs of colored image ([cd32f26](https://github.com/melMass/comfy_mtb/commit/cd32f26b167088d6b489e43b260c187ea5e4d223)) by [@ScottNealon](https://github.com/ScottNealon) in [#147](https://github.com/melMass/comfy_mtb/pull/147)
- 📝 adds a way to not load the imagefeed ([501c330](https://github.com/melMass/comfy_mtb/commit/501c3301056b2851555cccd75ab3ff15b1ab8e0c))
- 🐛 colored image mask input ([30c4311](https://github.com/melMass/comfy_mtb/commit/30c4311b69f6481a34f968cb67a9b5ce5d2e9fda))
- 🐛 handle font cache errors ([c43a661](https://github.com/melMass/comfy_mtb/commit/c43a661ba31dcd7720b4f32d8e96760e6191fbd9))
- 💄 register the COLOR type even for external extensions ([12b134a](https://github.com/melMass/comfy_mtb/commit/12b134ab4c937c192aaf4a3667d9885dd4fe43ca))
- ✨ mask crop output ([59a361a](https://github.com/melMass/comfy_mtb/commit/59a361af5870b8ffc984c6680dd3282d3553dcf9))
- 🚑️ thread font loading ([e4da832](https://github.com/melMass/comfy_mtb/commit/e4da832b99bd640b72c31b67178a3168e3238fa0))
- 📦 changed way of creating bbox from mask ([14ee9e2](https://github.com/melMass/comfy_mtb/commit/14ee9e23c009ab55fa3b2fc6ec60fb683c46d57d)) by [@Yurchikian](https://github.com/Yurchikian) in [#124](https://github.com/melMass/comfy_mtb/pull/124)
- ✨ expose invert of bboxfrommask ([53cb503](https://github.com/melMass/comfy_mtb/commit/53cb503866da6d83b47eaeb8073039ace2ae0a95))
- ✨ less strict csv parsing ([d5c4c5f](https://github.com/melMass/comfy_mtb/commit/d5c4c5f2649ecdb4bf7b517c5b33bbf8df753047))
- 🐛 fit number regression ([c8658df](https://github.com/melMass/comfy_mtb/commit/c8658dfbdd3a0ca8c3e88cd1adfddc55c7444045))
- 🐛 remove uneeded installs ([4e07450](https://github.com/melMass/comfy_mtb/commit/4e07450bcabb0105b5610e52f7d4692ea07f9c1d))
- 🐛 import issue ([255ac03](https://github.com/melMass/comfy_mtb/commit/255ac036bab1d776301857843d0e7a85e9a9dcb8))
- 🐛 wrong output for bbox ([8d12b59](https://github.com/melMass/comfy_mtb/commit/8d12b59844958fbc696d01d51162f97262664ae9))
- 🚑️ fallback when symlink detection fails ([278f22c](https://github.com/melMass/comfy_mtb/commit/278f22c2093b6eca63d2d00f7936774918707e4e))
- ✨ handle malformed styles.csv ([e6f6502](https://github.com/melMass/comfy_mtb/commit/e6f65026735770df8aced4a3acb75550ff1c84da))
- 🐛 encoding ([5af2840](https://github.com/melMass/comfy_mtb/commit/5af284067c65042bcdfff04a5d5a2360bf9e4af7))
- ⚡️ add the cli deps ([bb90e04](https://github.com/melMass/comfy_mtb/commit/bb90e0415f6a1ececbf468815dc0f5959d9a34e8))
- 🚑️ check for symlink ([25b933c](https://github.com/melMass/comfy_mtb/commit/25b933c698b250a411549d2600fae49bec225b7a))
- 🚑️ remove problematic dependencies ([5dfea51](https://github.com/melMass/comfy_mtb/commit/5dfea51dd8db2a4829e559eadeda22374b51c8a4))
- 🐛 batch support ([f1ff9fc](https://github.com/melMass/comfy_mtb/commit/f1ff9fc7c4684ad673c3178df3b8142dcf0b16ac))
- 🐛 automatically disable tiling if seamless is on ([4605f74](https://github.com/melMass/comfy_mtb/commit/4605f74f370d4d221ab1d50f21b72910fa6909c7))
- 🐛 debug node ([dc500b7](https://github.com/melMass/comfy_mtb/commit/dc500b788e885205f017956da6a71a677f822941))
- ⚡️ hack to handle prompt validation ([d49b257](https://github.com/melMass/comfy_mtb/commit/d49b2578c247dcba9b09b374d99f5cc45cac172d))
- ✨ deepbump update ([87b245c](https://github.com/melMass/comfy_mtb/commit/87b245c6a6895490e3612b235879fa90b62dea2b))
- 👷 user folder_paths to retrieve comfy root ([38df58a](https://github.com/melMass/comfy_mtb/commit/38df58a78c363ef2657011893d4d811676b1c664))
- 🐛 typo ([90aee83](https://github.com/melMass/comfy_mtb/commit/90aee83797a863cf4797cdbe187f949061cbd176))
- 🐛 do not resolve symlink for "here" ([a50b11b](https://github.com/melMass/comfy_mtb/commit/a50b11bdaa66f4e805811b1676c937ade11318c2))
- ✏️ use Union to allow support for <3.10 ([88a2779](https://github.com/melMass/comfy_mtb/commit/88a277968745ac990406b14d300a8ada9c575b11)) by [@M1kep](https://github.com/M1kep) in [#91](https://github.com/melMass/comfy_mtb/pull/91)
- ⚡️ simplify widgets cleanup ([cdd098e](https://github.com/melMass/comfy_mtb/commit/cdd098e10258401402b8023c9143532cfa4a1745))
- ✨ don't assume the install was ran ([cc43654](https://github.com/melMass/comfy_mtb/commit/cc43654af2987bc8860557caa99cde91e8309b21))
- 🐛 install ([616b2bf](https://github.com/melMass/comfy_mtb/commit/616b2bfc6c629cef1d30cb0d717bd805c3a086aa))
- 🐛 properly escape paths ([22cac9b](https://github.com/melMass/comfy_mtb/commit/22cac9b2d95910197941b73e7548735470bd3b17))
- 🐛 use relative paths in JS ([e2773ff](https://github.com/melMass/comfy_mtb/commit/e2773ff22e43e7756ad618344a03d661a576cf35))
- 💄 BatchFromHistory when "listening" ([3b07984](https://github.com/melMass/comfy_mtb/commit/3b07984716402fbbf5da41020bf73befd52e7ebf))
- ✨ save gif widget removal ([fe8f519](https://github.com/melMass/comfy_mtb/commit/fe8f519f8860b0610d8cafcd9b843b4171c2b3d4))

### Documentation

- 📚 update the wiki ([fa3199b](https://github.com/melMass/comfy_mtb/commit/fa3199be2b87bf3cb7484a0fee32a8ac099adc65))
- 📚 update wiki submodule ([49cea8d](https://github.com/melMass/comfy_mtb/commit/49cea8d94508b27781506e3b5509c65e1d84e80f))
- 📚 add the wiki as a submodule ([5998924](https://github.com/melMass/comfy_mtb/commit/59989249260a9c579ec851c50534b58f3f02cd61))
- 📚 missing doc ([c9836a8](https://github.com/melMass/comfy_mtb/commit/c9836a87f6823db1d53e56997417f3cbe8cc4727))
- 📚 use flat icon ([991af4f](https://github.com/melMass/comfy_mtb/commit/991af4f45ff8c660b2c45466bb219186699170ed))
- 📚 add banodoco channel link ([9ce34b4](https://github.com/melMass/comfy_mtb/commit/9ce34b47fd99b18db7997ccce44e6063f00b6801))
- 📚 udpate changelog ([8221c49](https://github.com/melMass/comfy_mtb/commit/8221c49942bd87c14d5063066315a449a1fee86e))
- 📝 add changelog ([0d817bf](https://github.com/melMass/comfy_mtb/commit/0d817bf326b4a22e2221264a414af50c3b7048b9))
- 📄 add note+ screenshot ([90d9636](https://github.com/melMass/comfy_mtb/commit/90d96366c8b7637b55d1b4f88cb9aca217c1414b))
- 📝 add cover image ([6b993b8](https://github.com/melMass/comfy_mtb/commit/6b993b84071bbb80ba1b8bd63576f31e35d05590))
- 📝 fix image size ([3e8c2fe](https://github.com/melMass/comfy_mtb/commit/3e8c2fe789925e7017c2f8c8d9164c139588aba4))
- 📝 add image ([3e93ea6](https://github.com/melMass/comfy_mtb/commit/3e93ea6f2c73353891b1a3f6223b5730bc69df37))
- 📝 explain optional nodes ([cea0b08](https://github.com/melMass/comfy_mtb/commit/cea0b08eb044756ab1b408f630435095b8969d36))
- 📝 add the example previews from the wiki ([8f90986](https://github.com/melMass/comfy_mtb/commit/8f909864bfaa9f2d0fbdcf3942eacb9d78ee8fb8))
- 📝 update node list ([4917e31](https://github.com/melMass/comfy_mtb/commit/4917e31c427c74d28c830fd7b2423cab393ba0f8))
- 📝 add some deprecation warnings and recommendations ([e11df9d](https://github.com/melMass/comfy_mtb/commit/e11df9d45c81d93f4334841de036b4aa3364375a))
- 📝 add a reference to SlickComfy for colab ([bb35098](https://github.com/melMass/comfy_mtb/commit/bb35098c656b0b2d30909b83df0a3b65c5975f78))

### Features

- ✨ add ModelPruner (wip) ([43d65ae](https://github.com/melMass/comfy_mtb/commit/43d65ae68c97e077117b17b7c9d1936583f965eb))
- ✨ Use dynamic contrast in Color Correct ([6abac2e](https://github.com/melMass/comfy_mtb/commit/6abac2e4706a3d937420213e01468bae10cc2017)) by [@christian-byrne](https://github.com/christian-byrne) in [#180](https://github.com/melMass/comfy_mtb/pull/180)
- ✨ StackImages add support for batch mismatch ([5060c56](https://github.com/melMass/comfy_mtb/commit/5060c561353e43624ec164cb73fce7d1d422f765))
- ✨ add BatchFloatMath ([f9d2ebf](https://github.com/melMass/comfy_mtb/commit/f9d2ebf91d09fc214fecf7501a5490b33c30aca2))
- ✨ add FLOATS to INTS ([1b7ae27](https://github.com/melMass/comfy_mtb/commit/1b7ae27cc1907bfba3c5166ec2c61547babd2e0a))
- ✨ debug dict ([63ee25d](https://github.com/melMass/comfy_mtb/commit/63ee25d001d4c94aa95dc8b39008f5d943f2ab45))
- ✨ add Swap BG/FG color menu item ([1caf7c1](https://github.com/melMass/comfy_mtb/commit/1caf7c18c372651b2be7227eb77e2251d963693d))
- ✨ BatchFloatFit the batch version of FitNumber ([ab58c36](https://github.com/melMass/comfy_mtb/commit/ab58c362124f0f4b3178534ca78cb924fb881534))
- ✨ add FloatToFloats (the counterpart) ([78a86da](https://github.com/melMass/comfy_mtb/commit/78a86daaf71dab5be34b90b13491460854718485))
- ✨ add some FLOATS batch nodes ([2159395](https://github.com/melMass/comfy_mtb/commit/2159395389429c5f7012e660b41fad48d376b39f))
- ✨ poc of the doc widget idea ([fac7529](https://github.com/melMass/comfy_mtb/commit/fac7529d1f7b6fc4b3b2e7f6022ebb23ec71169d))
- ✨ add the backend node for Constant ([dff5b22](https://github.com/melMass/comfy_mtb/commit/dff5b2201d73c1a91d4b5864e3b974e68846a011))
- ✨ add Constant node ([cbb5dd2](https://github.com/melMass/comfy_mtb/commit/cbb5dd2cf810d5648a64eae370dba610336b99d5))
- ✨ add FloatsToFloat ([6ebecfd](https://github.com/melMass/comfy_mtb/commit/6ebecfd8cf1dc3779384e565a65baa9dceb43660))
- ✨ add AutoPanEquilateral ([3513937](https://github.com/melMass/comfy_mtb/commit/35139371e84d715423015e05d1b4a6c1d88b0eb5))
- ✨ add MatchDimensions ([5db3ebe](https://github.com/melMass/comfy_mtb/commit/5db3ebedb9d38470c82544e45970775193add05c))
- ✨ add equilateral example ([8d65556](https://github.com/melMass/comfy_mtb/commit/8d65556c37f33d1c496504db92574805916dd613))
- ✨ enhance tiling tools ([ba73fc6](https://github.com/melMass/comfy_mtb/commit/ba73fc6af7039a4629a73cdc36a8c8736dc27c9d))
- ✨ add FLOATS support to blur ([92c810c](https://github.com/melMass/comfy_mtb/commit/92c810c5036f7a2b3f84a3fde8c81e6a2b046b07))
- ✨ add "tube" to Batch Shape ([f658fc3](https://github.com/melMass/comfy_mtb/commit/f658fc31e040141209384d98dfe84b766fe4ae11))
- ✨ note+ editor themes ([133da70](https://github.com/melMass/comfy_mtb/commit/133da705c94af2dfb3d2f38c0d9c2723c72cacf7))
- ✨ add ffmpeg gif export ([1b29aad](https://github.com/melMass/comfy_mtb/commit/1b29aad360116e631b7b4d34e98a5a631f134977)) by [@huanggou666](https://github.com/huanggou666) in [#159](https://github.com/melMass/comfy_mtb/pull/159)
- ✨ add "To Device" ([c28181f](https://github.com/melMass/comfy_mtb/commit/c28181f1615d2e183767aa76cc2350934330e546))
- ✨ add note+ example ([90f3bc2](https://github.com/melMass/comfy_mtb/commit/90f3bc2d953b299ea34e9e3a925f1a824b488855))
- 💄 node+ improvements ([4b29395](https://github.com/melMass/comfy_mtb/commit/4b29395000254382882c0d1be115b2ed80cd7c99))
- 📝 add note plus ([605c8db](https://github.com/melMass/comfy_mtb/commit/605c8db320e1531c6347f6888606fa50d8eb268b))
- 🚧 add playlist nodes ([cf96572](https://github.com/melMass/comfy_mtb/commit/cf965727e8e7064328704d88cd0410c61f1e686e))
- 🚨 add missing node ([16c1a59](https://github.com/melMass/comfy_mtb/commit/16c1a59312b1d9841f5f8a814eff93a1ddf04edb))
- ✨ Math Expression node ([142624e](https://github.com/melMass/comfy_mtb/commit/142624eea616a5622387b1b641c02605455ee6f1))
- 🚀 add optional inputs to colored image ([049983d](https://github.com/melMass/comfy_mtb/commit/049983dbe2dbce6b772908468c4042d2bfde5eb2))
- ✨ Add support for extra_model_paths.yaml ([d7b8ac8](https://github.com/melMass/comfy_mtb/commit/d7b8ac8e0c98b0d7a2e21889d35aad9f6b093560))
- ✨ add batch shake ([af94203](https://github.com/melMass/comfy_mtb/commit/af94203d1b461d934ca1c44211ca0f71a5d05d48))
- ✨ enhance concat images ([a798eb0](https://github.com/melMass/comfy_mtb/commit/a798eb07d0d891cfbd47013b442ef2fa3d7cc5bc))
- 💄 add a few more batch nodes ([c1d42de](https://github.com/melMass/comfy_mtb/commit/c1d42de0fcde86d2a167fb4b5e781ee987814da2))
- ✨ Batch node utilities ([cef5023](https://github.com/melMass/comfy_mtb/commit/cef5023efc17366a2e937ef43944de3587707fac))
- 🚨 Image Stack node (horizontal and vertical stack) ([bb3277d](https://github.com/melMass/comfy_mtb/commit/bb3277d85f4ca21735cb1f5237cb1430db88c183))
- 🚀 add seamless model hack ([21acc87](https://github.com/melMass/comfy_mtb/commit/21acc87ff0a84b7588f4b5aae0aeb5ae94bbbfbe))
- 🔧 debug handle a few more types ([638498c](https://github.com/melMass/comfy_mtb/commit/638498c6b47c2b2cab82f76aec1f3d46df67f263))
- 🎨 Add an editor for the styles loader ([2faa2f2](https://github.com/melMass/comfy_mtb/commit/2faa2f2a148a4dbf5525e4945f688a239f244546))
- ✨ add a static assets path ([6a00d1d](https://github.com/melMass/comfy_mtb/commit/6a00d1da5a8a5fa47af1bf1ab5d3cd206c599841))
- ✨ add  Interpolate Clip Sequential ([a71c273](https://github.com/melMass/comfy_mtb/commit/a71c273baf450ad7e2a7e032451f015d3be3e9e9))

### Miscellaneous Tasks

- 🧹 add fields for the registry ([bb5682a](https://github.com/melMass/comfy_mtb/commit/bb5682aa6da923859db33830c2e46f24b19199a1))
- 🧹 add pre-commit ([59612fd](https://github.com/melMass/comfy_mtb/commit/59612fd8110a888f0081433242a2b5a5f7e46da6))
- 🧹 migrate from poetry to setuptools ([dfd17f6](https://github.com/melMass/comfy_mtb/commit/dfd17f6d783e784df7dab38d185c747b4c04d1d0))
- 🧹 remove logs ([1070edd](https://github.com/melMass/comfy_mtb/commit/1070edd0245fb235183d5f38cd1bebf6e0405f97))
- 🧹 add more pyproject meta ([644371e](https://github.com/melMass/comfy_mtb/commit/644371e5b5a2b8260fc5c6f699465b0bc1c81d57))
- 🤖 move at the proper location ([f3d468c](https://github.com/melMass/comfy_mtb/commit/f3d468cfc238f13905a13a7b2225e3711129c64d))
- 🤖 add CI to publish to ComfyUI Registry ([6cd448b](https://github.com/melMass/comfy_mtb/commit/6cd448b026956cdf3f1b81e93724b295316fbf09)) by [@haohaocreates](https://github.com/haohaocreates) in [#182](https://github.com/melMass/comfy_mtb/pull/182)
- 🧹 add ComfyUI registry to pyproject.toml ([5951c90](https://github.com/melMass/comfy_mtb/commit/5951c90b10f9b77b2b617e83efe0112f43c8daef)) by [@haohaocreates](https://github.com/haohaocreates) in [#181](https://github.com/melMass/comfy_mtb/pull/181)
- 🧹 update types ([96a0da9](https://github.com/melMass/comfy_mtb/commit/96a0da9dbd051d1fcf8b332c54ed2d307d8ae0dd))
- 🧹 use a gettattr fallback ([a344cdc](https://github.com/melMass/comfy_mtb/commit/a344cdcba9823ca1fb0762795068039b1e1cf0ab))
- 🧹 cleanup js ([64cc4e9](https://github.com/melMass/comfy_mtb/commit/64cc4e9649853023d645245bea1e1ceb11073f01))
- 🧹 add savedatabundle js part ([edd7c3f](https://github.com/melMass/comfy_mtb/commit/edd7c3f5d075b640e9cdb067ebfe51c42ff61791))
- 🧹 wip dynamic multitype ([71bfdd6](https://github.com/melMass/comfy_mtb/commit/71bfdd61d731ce15f9bd0bb19d65b5af208d5dcf))
- 🧹 applied some linting ([fe49312](https://github.com/melMass/comfy_mtb/commit/fe49312cbef03c6540304448fa88aa7a88391efa))
- 📝 header links not parsed ([514c0d2](https://github.com/melMass/comfy_mtb/commit/514c0d2eda9990435eb18258d4bbd1aa137feb3d))
- 📝 hardcode links in changelog ([915b744](https://github.com/melMass/comfy_mtb/commit/915b7444a9db83f349d83b636304af0d276f529f))
- 🔖 local updates ([6c5e5d3](https://github.com/melMass/comfy_mtb/commit/6c5e5d36379bdab223b4503e42b7956b55a82ab0))
- 📝 update node list ([dd27f99](https://github.com/melMass/comfy_mtb/commit/dd27f990c72fa94aff205eb314a8ea360f57479e))
- ✨ update node_list ([537a0d8](https://github.com/melMass/comfy_mtb/commit/537a0d8108d0caa3ab2daeafd1d25d680214ef26))
- ✨ local stuff ([9afad1a](https://github.com/melMass/comfy_mtb/commit/9afad1a1680073006d946be10f8c97b75ddfe253))
- 📝 fix update issue template ([da290db](https://github.com/melMass/comfy_mtb/commit/da290dbcf2952a56be9334f7bf9dc4d8fa64a21d))
- 📝 update issue template ([b949bb4](https://github.com/melMass/comfy_mtb/commit/b949bb406bc1929634600465ea389eaedefe6e6f))

### Refactor

- ⚡️ small local fixes ([bcac665](https://github.com/melMass/comfy_mtb/commit/bcac66508d2e788cc437da289d1ccede19465b8c))
- 🗑️ remove unused code in install script ([5b75436](https://github.com/melMass/comfy_mtb/commit/5b75436610c6312adf47c6baa3e9fe9cc7d56dcf))

### Merge

- 🔀 pull request #109 from melMass/dev/0.2.0 ([87e301d](https://github.com/melMass/comfy_mtb/commit/87e301d120a542d5aabe544bec10d38dbd19b2f6)) in [#109](https://github.com/melMass/comfy_mtb/pull/109)
- 🔀 pull request #86 from melMass/feature/styles-editor ([cbdb816](https://github.com/melMass/comfy_mtb/commit/cbdb816164900061ddaa1671f4287763d0b79ee1)) in [#86](https://github.com/melMass/comfy_mtb/pull/86)

### Wip

- 🚧 curve widget logic fixed ([e312b02](https://github.com/melMass/comfy_mtb/commit/e312b02ad2f8334e87654a20b0114837df229371))
- 🚧 dump3 ([eedbb4b](https://github.com/melMass/comfy_mtb/commit/eedbb4bc6581bef85c746307fe9d53360ea45bcf))
- 🚧 dump ([fa23975](https://github.com/melMass/comfy_mtb/commit/fa2397585fff4f54bcf17f0b0e0083c427b34fa8))
- 🚧 dump ([0d0fb8e](https://github.com/melMass/comfy_mtb/commit/0d0fb8e13a5da54a44a96a04607f7a349f8fdb03))
- 🚧 add text template node ([af2175a](https://github.com/melMass/comfy_mtb/commit/af2175a1fc0c2fb29ef3493f242fe45ec6fcabac))

## New Contributors
* [@haohaocreates](https://github.com/haohaocreates) made their first contribution in [#182](https://github.com/melMass/comfy_mtb/pull/182)
* [@vxkj1211](https://github.com/vxkj1211) made their first contribution in [#177](https://github.com/melMass/comfy_mtb/pull/177)
* [@huanggou666](https://github.com/huanggou666) made their first contribution in [#159](https://github.com/melMass/comfy_mtb/pull/159)
* [@hongminpark](https://github.com/hongminpark) made their first contribution in [#154](https://github.com/melMass/comfy_mtb/pull/154)
* [@ScottNealon](https://github.com/ScottNealon) made their first contribution in [#147](https://github.com/melMass/comfy_mtb/pull/147)
* [@Yurchikian](https://github.com/Yurchikian) made their first contribution in [#124](https://github.com/melMass/comfy_mtb/pull/124)
* [@M1kep](https://github.com/M1kep) made their first contribution in [#91](https://github.com/melMass/comfy_mtb/pull/91)
## [0.1.4] - 2023-08-12

### Bug Fixes

- 🚀 pending fixes ([ea5d73d](https://github.com/melMass/comfy_mtb/commit/ea5d73d48cfa4046f48a52609cff7f754d8364ed))
- 🚑️ image resize infinite loop ([30d6cfe](https://github.com/melMass/comfy_mtb/commit/30d6cfe81292d0f7702544b3c2cbad1820c4a926))
- ✨ update example files ([610afe0](https://github.com/melMass/comfy_mtb/commit/610afe031f21d737b2fd5128e4be7100b6666181))
- 🐛 simplify install steps ([4fc84d6](https://github.com/melMass/comfy_mtb/commit/4fc84d615dd0f546442c3537f00c52366db4ca9b))
- ✨ refactor ([8523392](https://github.com/melMass/comfy_mtb/commit/8523392df74c586dc940841ddbb5069943b16f7d))
- 🐛 debug rgba ([40560f8](https://github.com/melMass/comfy_mtb/commit/40560f8154d3ddeabf708be4d111370648d466ac))
- 🎨 rename fun to generate ([e7f72f9](https://github.com/melMass/comfy_mtb/commit/e7f72f9825da58254e3084b4ba91f76e6cf2cf5f))
- ✨ refactor existing ([1144466](https://github.com/melMass/comfy_mtb/commit/11444662b9198861b62aff06a08b9c9ea01dd8bd))
- ⚡️ move getbatchfromhistory to graphutils ([2eccba4](https://github.com/melMass/comfy_mtb/commit/2eccba4e33b21d1d080cb2f415f76a93488120f0))
- 🚧 wip dependency installer UI ([630b492](https://github.com/melMass/comfy_mtb/commit/630b492347f75d7308b31a000061b41d7dfa4a10))
- 🐛 image feed zorder ([0fb2d4d](https://github.com/melMass/comfy_mtb/commit/0fb2d4da90a7e65f82b3f9c8942a68e360456cf7))
- ⬇️ download_antelopev2 ([4dd5321](https://github.com/melMass/comfy_mtb/commit/4dd532185223a1fa5978446e7bb75d32d77ebdb5))
- 🚑️ frontend pushed too early ([91f60d4](https://github.com/melMass/comfy_mtb/commit/91f60d4c463c474ac10e868e8e73e13fa019856b))
- 🚑️ missing input ([84ac8ac](https://github.com/melMass/comfy_mtb/commit/84ac8ac852aeb962029bfd8369fe5ed59a203977))
- 🐛 shell command bug ([3d5075f](https://github.com/melMass/comfy_mtb/commit/3d5075fea2e219a179271c9810017c7e38bff6cc))
- 🚑️ remove pipe mode from the install.py ([b854a30](https://github.com/melMass/comfy_mtb/commit/b854a302ce4708d2ad2dac249860308dbdcae5a6))
- ⚡️ colab install ([36d8e6b](https://github.com/melMass/comfy_mtb/commit/36d8e6bdb06edab72ccfb686266d2e644a9f028c))
- 🚑️ install typo ([ffa1a87](https://github.com/melMass/comfy_mtb/commit/ffa1a87b9184df5a3699a6118714b39d359bde4d))

### Documentation

- 📝 link the actual action instead of badge ([098d74a](https://github.com/melMass/comfy_mtb/commit/098d74a3cd8449d836569a074995e20d775c6728))
- 📝 add action badge ([e74314b](https://github.com/melMass/comfy_mtb/commit/e74314b04eb218c140482ccf704b61af06db3f4d))

### Features

- 💫 export to prores -> export with ffmpeg ([a4d99d9](https://github.com/melMass/comfy_mtb/commit/a4d99d966b1207191243a9749385b998d1a9c6b1))
- 🔥 add any to string & refactor ([dbdb872](https://github.com/melMass/comfy_mtb/commit/dbdb872b74e18c16feb44bd037abc3aafbb4700f))
- ✨ add UI for interpolate clip sequential ([5ec5511](https://github.com/melMass/comfy_mtb/commit/5ec551143302b2a94ca82e477f684ecee23f1459))
- ✨ add portable reqs ([3f14b16](https://github.com/melMass/comfy_mtb/commit/3f14b1676d28f5ffa1f47fda00b9bc244951045c))
- ✨ add border extension ([fb64484](https://github.com/melMass/comfy_mtb/commit/fb644847ca434123e8e8e4991d33949fd31e3cbe))
- ✨ use PIL for gif saving ([2bc7ae8](https://github.com/melMass/comfy_mtb/commit/2bc7ae88bf4cdfa575d11233c0e6f7b07f9dfd23))
- 🎨 update node list ([a54d7d5](https://github.com/melMass/comfy_mtb/commit/a54d7d5346c272898dd4e67c65495de7325ab3a0))
- ✨ install fix ([512de60](https://github.com/melMass/comfy_mtb/commit/512de6023e55f2cc47516bf44436efe22157273f)) in [#41](https://github.com/melMass/comfy_mtb/pull/41)

### Miscellaneous Tasks

- 💄 encoding ([49c64c7](https://github.com/melMass/comfy_mtb/commit/49c64c74eb3e99f456b563bbd79e3fe47a85c70d))
- 🚀 only fetch controlnet_preprocessor deps ([414beb9](https://github.com/melMass/comfy_mtb/commit/414beb99a1f9bf719eca6ac139c9b2ccdfd6d743))
- 🚀 add controlnetpreprocessors to tests ([63b3aec](https://github.com/melMass/comfy_mtb/commit/63b3aece2ba05adc2b655afeb41e3d47e7887b33))
- ✨ remove unused input ([d4f791d](https://github.com/melMass/comfy_mtb/commit/d4f791d7a14ba9cb8abd7c95ba70b081fee5fb7c))
- ✨ use the same cwd as manager ([2ff0467](https://github.com/melMass/comfy_mtb/commit/2ff04672daff773d52e1552dca1bf616bc32daa6))
- 🎨 no brace glob ([bbfcb62](https://github.com/melMass/comfy_mtb/commit/bbfcb62c398de39058bcb6e18161425059d53e8e))
- 🎨 extract txt ([a22fd01](https://github.com/melMass/comfy_mtb/commit/a22fd01d664276e4cd833ae1326feeece1d1deaf))
- 🎨 also push wheels_order to releases ([8e5b776](https://github.com/melMass/comfy_mtb/commit/8e5b7765cc0c6730bd5517ccfd56e817ea39bd3a))
- 🚧 more info for bug reports ([3dadc11](https://github.com/melMass/comfy_mtb/commit/3dadc119f44fca1029ec4b349d71ce99fb20a4b6))
- ✨ individual wheels ([346ff64](https://github.com/melMass/comfy_mtb/commit/346ff649d50c9f0286ad2243938406fefb62853b))

### Refactor

- 🚧 tidy ([4f30829](https://github.com/melMass/comfy_mtb/commit/4f30829e06c41b3685644bfe7bece07e0bcfb70e))
- ♻️ get batch from history ([13d255a](https://github.com/melMass/comfy_mtb/commit/13d255a730b08c4903647875350b9b3dcd61b4a6))

### Revert

- 💄 use BOOLEAN instead of BOOL ([cfb3b23](https://github.com/melMass/comfy_mtb/commit/cfb3b237cf64b512414a17f71e6d89c3355aa8ef))

### Testing

- 🧪 remove sha input ([c5bbe83](https://github.com/melMass/comfy_mtb/commit/c5bbe83008bb194cbd6ad5e3dc70cb3850b18985))
- 🧪 ci for comfy embedded ([7b3afca](https://github.com/melMass/comfy_mtb/commit/7b3afca8179760e35e8a6fbf742080dee13e4fc7))

### Merge

- 🔀 pull request #50 from melMass/dev/august-refactor ([2ecd470](https://github.com/melMass/comfy_mtb/commit/2ecd4700d77c0727e6b5d2124e0a6ebd48ec96ed)) in [#50](https://github.com/melMass/comfy_mtb/pull/50)

## [0.1.3] - 2023-07-29

### Bug Fixes

- 🔥 manage pip from install only, remove requirements.txt ([247fbfb](https://github.com/melMass/comfy_mtb/commit/247fbfbc216b8259d607e0699d5b990b6a06ca71)) in [#38](https://github.com/melMass/comfy_mtb/pull/38)
- 🎨 use image ratio for imagefeed ([f5cd56c](https://github.com/melMass/comfy_mtb/commit/f5cd56ce861c8c0a931744ae6cf2b96e9c8bca06))

### Documentation

- 📝 update imagefeed preview ([cbcacbe](https://github.com/melMass/comfy_mtb/commit/cbcacbe3c92ebb5f74d046b83504c3723710f130))
- 📝 fix typo and add more details ([7c020ba](https://github.com/melMass/comfy_mtb/commit/7c020bab288aa7d17dc937b5f102319d43c3ebb3))

### Miscellaneous Tasks

- ✨ use wheel order if present ([9b24edd](https://github.com/melMass/comfy_mtb/commit/9b24eddd9c51004af08d7ac6ff2b6473dd3ee161))
- ✨ store order of install for wheels ([5053142](https://github.com/melMass/comfy_mtb/commit/505314294f02e7c19ac95e4d0ed37fd397a54b46))

## [0.1.2] - 2023-07-28

### Bug Fixes

- ✨ various small things ([0e311cf](https://github.com/melMass/comfy_mtb/commit/0e311cf2c64cf2b4861d4cc612a3409390e3039a))
- 📝 last release ([889f08c](https://github.com/melMass/comfy_mtb/commit/889f08c08b721be8fdb4e4d7eacc47169d5692d6)) in [#36](https://github.com/melMass/comfy_mtb/pull/36)
- 📝 narrow requirements ([5d661b2](https://github.com/melMass/comfy_mtb/commit/5d661b2509fecf3940c3c0fab25b16ec0eae7a2d))
- ✨ Separate FaceAnalysis model loading ([d143e83](https://github.com/melMass/comfy_mtb/commit/d143e83dba3bffa16e1b98d7ad1e9cf92dc94db2))
- ⚡️ update examples to match wiki ([3dfe98c](https://github.com/melMass/comfy_mtb/commit/3dfe98c7957df48723380de85e1242a424ec23de))

### Documentation

- 📝 add readme for web extensions features ([be162a2](https://github.com/melMass/comfy_mtb/commit/be162a20477258627fa0d742c97a478bd085ff4f))
- 📝 link to the proper lang instructions ([232cf89](https://github.com/melMass/comfy_mtb/commit/232cf8966cc20291b60c68f487dfd37bf6aa4dfa)) in [#33](https://github.com/melMass/comfy_mtb/pull/33)
- 📝 update readmes ([96a0618](https://github.com/melMass/comfy_mtb/commit/96a0618c5990a8559a9e2dd17c868d3465b8ca90))

### Miscellaneous Tasks

- 🎉 bump version ([9e751a2](https://github.com/melMass/comfy_mtb/commit/9e751a242f4e9afee3dc5c871c414b29b9706ff6))
- 👷 remove stale example ([c237737](https://github.com/melMass/comfy_mtb/commit/c2377374201fc34b107c8b7db1cdeb2f483d1e18))
- 🐛 fix size ([c0cc557](https://github.com/melMass/comfy_mtb/commit/c0cc5572d8c727568eca8a3d0f116a1f540c31ff))

## [0.1.1] - 2023-07-24

### Bug Fixes

- 🎨 improve a bit the HTML response of endpoints ([50d51c7](https://github.com/melMass/comfy_mtb/commit/50d51c70d04e49e9df524975c171288c0fc0b20f))
- 🐛 caching issues ([55c9736](https://github.com/melMass/comfy_mtb/commit/55c9736a9b2ca036926be4b06406121bfb9ebad2))
- 🔥 remove notice ([abf1e82](https://github.com/melMass/comfy_mtb/commit/abf1e82adb9fac8cd70d5c409baad55309ef6fe1))
- 🔥 use BOOL everywhere ([a393793](https://github.com/melMass/comfy_mtb/commit/a393793cfa93721eac46295723076a1dda940dcd))

### Documentation

- 📝 added lang links ([bbdac97](https://github.com/melMass/comfy_mtb/commit/bbdac97e49af4e90d22eeec3f63b96ecc126ffcf))
- 📝 add comfyforum example ([10d0503](https://github.com/melMass/comfy_mtb/commit/10d05031b1791ab3534cf838be6eb75df638dfb6))

### Features

- 🚧 jupyter seems to require an __init__ there ([9a4eda3](https://github.com/melMass/comfy_mtb/commit/9a4eda3ef573bf382c13515f67ae8a415bf61abd))
- ⚡️ use notify ([a2ecc11](https://github.com/melMass/comfy_mtb/commit/a2ecc11ebde79c2403959bf09c258f3a2465894a))
- ✨ first version of Notify ([7e9c97e](https://github.com/melMass/comfy_mtb/commit/7e9c97ecb48672b25e5ed17b9b35dba9208ac311))
- ⚡️ add an "actions" endpoint ([3de160a](https://github.com/melMass/comfy_mtb/commit/3de160af25b516c02aaa8cc32baec16e9ef358fb))
- ✨ add Unsplash Image node ([8d3cc39](https://github.com/melMass/comfy_mtb/commit/8d3cc39b72dff1b5eb61bf7e2e395753c138ec8a))
- ✨ add back Save Tensors ([7142b28](https://github.com/melMass/comfy_mtb/commit/7142b284adc7fba9a1bdafd1a52621bfc168bde1))
- ✨ add TransformImage node ([11128ff](https://github.com/melMass/comfy_mtb/commit/11128ff85a7e0b4a54f405548969c2478da26df6))

### Miscellaneous Tasks

- 🚀 bump version ([cf86552](https://github.com/melMass/comfy_mtb/commit/cf865529ab64b350cd7af964b41160e7d130d12d))
- 🚀 Remove large files from release ([3b9190a](https://github.com/melMass/comfy_mtb/commit/3b9190a69b002b8933c097fd6655bb4fe07264d2))

### Refactor

- ✨ cleaned up frontend code a bit ([3801a44](https://github.com/melMass/comfy_mtb/commit/3801a443bc1e89c70fdb35ce0b1724d86fa22928))
- ⚡️ remove empty inits ([21729b2](https://github.com/melMass/comfy_mtb/commit/21729b2784a50fcaf24a63ac283bdae475a53ce7))

### Merge

- 🔀 pull request #32 from melMass/dev/next ([8695cd3](https://github.com/melMass/comfy_mtb/commit/8695cd3f1b6d27b5cd6c616ed1215ea2f25c5304)) in [#32](https://github.com/melMass/comfy_mtb/pull/32)

## [0.1.0] - 2023-07-22

### Bug Fixes

- 🔥 properly match built wheels ([119b4d6](https://github.com/melMass/comfy_mtb/commit/119b4d6e16c2a90db1664ccaac748507feb73ea0)) in [#30](https://github.com/melMass/comfy_mtb/pull/30)
- ✨ also try to copy web if symlink fails ([0df55de](https://github.com/melMass/comfy_mtb/commit/0df55def29fb992751010f6b8a707699f230ff37))
- ✨ install process tested in comfy-manager (embed, colab) ([b40730d](https://github.com/melMass/comfy_mtb/commit/b40730ddbc3f8e3e7d5a17e9e9e4526ff37977fd))
- 🚀 try to support remote install too ([3c66de2](https://github.com/melMass/comfy_mtb/commit/3c66de2500a89efd2d2e3af88fc58429af725789))
- 💄 save gif issues ([7335003](https://github.com/melMass/comfy_mtb/commit/7335003346e83666c5dee631b8e6b15586d871e7))
- 🚑️ always use latest for now ([fccf313](https://github.com/melMass/comfy_mtb/commit/fccf31348994ab6e344a1ab00a8f9998309f9319))
- 🐛 install logic ([7e301e2](https://github.com/melMass/comfy_mtb/commit/7e301e2a067d41cba9b8ef357496dd1df94e4cdd))
- 🎉 remove tests & add missing docs ([4e6b877](https://github.com/melMass/comfy_mtb/commit/4e6b87719989aa144946c5c9a43b9398c20bf11e))
- ⚡️ update node_list ([c794d6a](https://github.com/melMass/comfy_mtb/commit/c794d6a071778220d654b526d2edfddcc79752fc))
- 🚑️ set debug level from endpoint ([18402e3](https://github.com/melMass/comfy_mtb/commit/18402e3be1ab47e10109cfd2dff18863a1ee56f7))
- 🐛 add base64 prefix to outputs ([0950f99](https://github.com/melMass/comfy_mtb/commit/0950f9914c9bbed7c89f3de33a967cb76f9d0bbb))
- 🎨 refactor and add Gif preview on node ([c2e8379](https://github.com/melMass/comfy_mtb/commit/c2e83794faeb8da708c98908882e38b2a42827bd))
- ✨ Various widgets issues ([27500ca](https://github.com/melMass/comfy_mtb/commit/27500ca432d686774b991045b7cffc58c0b67faf))
- 🔥 deprecate some nodes and fix image list ([9aa934f](https://github.com/melMass/comfy_mtb/commit/9aa934f70ff6adf91efb26aa8e5cb21ec575196a))
- 🐛 crop nodes ([67d3783](https://github.com/melMass/comfy_mtb/commit/67d3783ac9186da6bba4b7dc7e8dc3d5db5a1b0f))
- 🐛 tensor2pil ([8a59508](https://github.com/melMass/comfy_mtb/commit/8a59508ff91d6b2d9ca287ef1c054ec5755337a4))
- ⚡️ a few missing __doc__ ([ab09cca](https://github.com/melMass/comfy_mtb/commit/ab09ccadd905bebbf1b7b2d992e96e36fc60d68a))
- ⚡️ from tensor2np always returning a list ([6168b3a](https://github.com/melMass/comfy_mtb/commit/6168b3a2ac38b5eebed3daf9e52df5742abf6813))
- 🚑️ TF by default fills vram ([c225da5](https://github.com/melMass/comfy_mtb/commit/c225da5f298acb4cb2b39022382543c0c966d428))
- ✨ leftovers ([da3e6f4](https://github.com/melMass/comfy_mtb/commit/da3e6f47c6073e73cf9d3a3cd23ba5ccbe1fedce))
- ✨ handle non fork gdown in model dll ([95797e8](https://github.com/melMass/comfy_mtb/commit/95797e823e12e62ae8758753f60c34afbe19ec90))
- ✨ properly add the submodules ([00510ed](https://github.com/melMass/comfy_mtb/commit/00510ed0b8583dd64518daa67d963582f4f029d3))
- 📌 remove sad talker for now ([1622cbc](https://github.com/melMass/comfy_mtb/commit/1622cbcb9d51ddd0e1a8b4d87ba47b99327163eb))
- 🎨 narrow requirements ([1a92ef7](https://github.com/melMass/comfy_mtb/commit/1a92ef734dd4271efc875856038f9e3b6b9ded6c))
- 🚀 use the comfy util to handle graph interruption ([9752f3e](https://github.com/melMass/comfy_mtb/commit/9752f3e9dec9aa59cfa809aa14f0151594c03858))
- 🔥 much faster (using GPU) on windows ([2f455aa](https://github.com/melMass/comfy_mtb/commit/2f455aaca55c0a044735c768b295d077b2f5b8d6))
- 🐛 uint8 to uint16 ([be5a655](https://github.com/melMass/comfy_mtb/commit/be5a655cfaba1794f7b09c82d65de42e2b031720))
- ✨ add missing requirements ([b779bc3](https://github.com/melMass/comfy_mtb/commit/b779bc39ac19f779aeb73c98b916671d1d16806f))
- 📝 don't propagate base logs ([7fd99c2](https://github.com/melMass/comfy_mtb/commit/7fd99c25c4e50566def5c5166a9d9059b1febfa6))
- 🐛 bg upscaler in gfpgan ([fee48ad](https://github.com/melMass/comfy_mtb/commit/fee48adff3d66960cb17836f3f4efbfd0c8740c4))
- 📝 separate debug / info better ([e24863d](https://github.com/melMass/comfy_mtb/commit/e24863d1f9f63f367a2b392e6228ffa42927b71b))
- 🔥 change log level of the base logger ([7538c2c](https://github.com/melMass/comfy_mtb/commit/7538c2c4bad8390a32226dc0a5a6ef978b00d201))
- ✨ handle externs dynamicly ([6ef308a](https://github.com/melMass/comfy_mtb/commit/6ef308a87062c91e2d7249c05d96c6fb76e5a6c4))
- 🐛 separate faceswap model load ([8e267c0](https://github.com/melMass/comfy_mtb/commit/8e267c0204ce5abe8e113fd401234d49f377646a))

### Documentation

- 📝 fold each comfy mode ([3c3c438](https://github.com/melMass/comfy_mtb/commit/3c3c4380bd1a3f0eed5216b835e076c26fce2f88))
- 📝 add more description to examples ([46eab5c](https://github.com/melMass/comfy_mtb/commit/46eab5ca2f0e04d872d87c849b11551fd219bdb9))
- 📝 add model notice ([cbe67ed](https://github.com/melMass/comfy_mtb/commit/cbe67edd4befb7260be01fa09af8448e5bcf5680))
- 📝 add preview for examples ([b5176ca](https://github.com/melMass/comfy_mtb/commit/b5176ca0ee489ada52b6632f68b794b4f709d5ba))
- 📝 add jp and cn (using deep translation) ([da559b9](https://github.com/melMass/comfy_mtb/commit/da559b9eaf135a49c0ab9bfa45573baf0c18dfb2))
- 📝 update readme ([b0fb522](https://github.com/melMass/comfy_mtb/commit/b0fb5222cb19e4004533d3367863be5c9ce8e72b)) in [#15](https://github.com/melMass/comfy_mtb/pull/15)
- 📝 update README.md ([f8dc768](https://github.com/melMass/comfy_mtb/commit/f8dc768635a2d21f6ff81b42c418724c432159bf))
- 📝 updated instructions ([c3b9fd4](https://github.com/melMass/comfy_mtb/commit/c3b9fd4afedbb46748aef17b40e167a4cfad65f5))

### Features

- ✨ update install instructions ([7be37db](https://github.com/melMass/comfy_mtb/commit/7be37dbbfac45e8038f94ced8a2fa8ec2b06fb34))
- 🚀 add install script ([dad3966](https://github.com/melMass/comfy_mtb/commit/dad3966ba219c1998e4fc7f6e641864fb0e7c3e8))
- 🚧 add my CLIs ([44eaae5](https://github.com/melMass/comfy_mtb/commit/44eaae5c79f4dbec344053d945e7275be5c3c0a5))
- ✨comfy_widget shared utils ([91bb95d](https://github.com/melMass/comfy_mtb/commit/91bb95da914468de533b040324700c7f9707e4fb))
- 🚀 debug node ([b27b8ef](https://github.com/melMass/comfy_mtb/commit/b27b8ef91fe7335b1df3766547edaf4b9625ae4d))
- ✨ add FitNumber node ([aa551eb](https://github.com/melMass/comfy_mtb/commit/aa551ebe57801c69010815119fe21e19a858780c))
- 🔥 add API endpoints ([95afbdb](https://github.com/melMass/comfy_mtb/commit/95afbdbf76e66897e632252d876384ada9acf153))
- ✨ categorize ([d2b3962](https://github.com/melMass/comfy_mtb/commit/d2b396236a10fe620ebebabd5a22c36159921913))
- 🚀 add a few examples ([b9c1d3d](https://github.com/melMass/comfy_mtb/commit/b9c1d3df7a1460fe9ffa84f6f9ea0cfb5409de1a))
- ✨ added a way to export the node list ([5f5297f](https://github.com/melMass/comfy_mtb/commit/5f5297f80debc77f3fda2f0d37b3acff8419140d))
- ✨ WIP batch from history ([cde7293](https://github.com/melMass/comfy_mtb/commit/cde72938d5ffd09179f5974676e12d4599a8d6ff))
- ✨ extract node names using ast ([38f6147](https://github.com/melMass/comfy_mtb/commit/38f61473bc23b4c5d4efc5048d54c059565a6fa0))
- 🔥 add batch support for load image sequence ([3faadc4](https://github.com/melMass/comfy_mtb/commit/3faadc4b8a5049cb8c264b8a3d50565adec405f1))
- 🎨 add support for image.size(0) == 0 ([629e2b5](https://github.com/melMass/comfy_mtb/commit/629e2b5f5fbebe4e79e8b7a4cff2de6017e79225))
- ✨ image feed ([99eb5ae](https://github.com/melMass/comfy_mtb/commit/99eb5ae0c7413f6ab1f24cfc8337c9b1b2d9824c))
- ✨ FILM interpolation nodes ([e04e77e](https://github.com/melMass/comfy_mtb/commit/e04e77eb097735ec1369dec51238cdcc5abe39b7))
- ✨ add an headless option for model downloads ([217e8a1](https://github.com/melMass/comfy_mtb/commit/217e8a1546d06b97250d99612ac6bdb5ce89e155))
- 🐛 support batch count > 1 for restore face ([8ef48a0](https://github.com/melMass/comfy_mtb/commit/8ef48a013a8d6b832b8c0c7dcabc1b78c27ff207))
- 🚧 wrapper for GFPGAN bg upscaler ([88cdcc6](https://github.com/melMass/comfy_mtb/commit/88cdcc6a87dae452924e8915eccdadc69d7d136e))
- ✨ add GFPGAN (FaceRestore) ([3a6e545](https://github.com/melMass/comfy_mtb/commit/3a6e5450502f3b1d7c505178fc9ba337cd95c39e))

### Miscellaneous Tasks

- ✨ before categorize ([0cc54e5](https://github.com/melMass/comfy_mtb/commit/0cc54e58ec86c28354cae37e14f39e831c13ea02))
- ✨ add more issue templates ([710a638](https://github.com/melMass/comfy_mtb/commit/710a638a8187ef08254478f684307dccdebcded2)) in [#25](https://github.com/melMass/comfy_mtb/pull/25)
- ✨ add bug report template ([f927bc7](https://github.com/melMass/comfy_mtb/commit/f927bc7c9a82951e6df4763433732f20ea87e9cb))
- 🍻 create FUNDING.yml ([f634fe0](https://github.com/melMass/comfy_mtb/commit/f634fe0e6b2db28138e4bd7932fbfc8606a0f033))
- 🍻 add bmc to readme ([cd1b603](https://github.com/melMass/comfy_mtb/commit/cd1b603565464fe98a718e1fbaa8c7cd84057576))
- 📝 extra files from another branch ([b78be8f](https://github.com/melMass/comfy_mtb/commit/b78be8fd3cd36666fd94a3ab08eca11cce526043))
- 🚀 push leftovers ([4c41fe7](https://github.com/melMass/comfy_mtb/commit/4c41fe7af9f8e16d895eb06223349e1294dd4698))

### Refactor

- ♻️ removes a few nodes, moved other around ([4d8ddac](https://github.com/melMass/comfy_mtb/commit/4d8ddaca320ce483640d030618e70730b3453df2))
- ♻️ remove test ([68c250e](https://github.com/melMass/comfy_mtb/commit/68c250e890dacae9f627b0d266ad6dcab0fa0c8b))
- 🚧 remove color_widget ([e480d07](https://github.com/melMass/comfy_mtb/commit/e480d071171cffa789930620f1e7ccc76473bf93))

### Testing

- 🔧 pipe detection ([ee17d57](https://github.com/melMass/comfy_mtb/commit/ee17d57c3d6d71fda1a5acc2cf85f936c525bc87))

### Install

- 🚧 handle symlink errors ([d982b69](https://github.com/melMass/comfy_mtb/commit/d982b69a58c05ccead9c49370764beaa4549992a))

### Merge

- 🔀 pull request #22 from melMass/dev/next-release ([c34de0a](https://github.com/melMass/comfy_mtb/commit/c34de0ab351b2c95d7fa4fab4487155bee6bfa3a)) in [#22](https://github.com/melMass/comfy_mtb/pull/22)
- 🎉 pull request #11 from dev/frame_interpolation ([1e28606](https://github.com/melMass/comfy_mtb/commit/1e28606427bcc8d895b87eaa6cd4147ab6d9a11f)) in [#11](https://github.com/melMass/comfy_mtb/pull/11)
- 🎉 pull request #8 from dev/small-fixes ([7585624](https://github.com/melMass/comfy_mtb/commit/7585624de5895eb34c6a520d4dab18b47e64b6ca)) in [#8](https://github.com/melMass/comfy_mtb/pull/8)

## [0.0.1] - 2023-06-28

### Bug Fixes

- 🤦 add missing file ([e2c4561](https://github.com/melMass/comfy_mtb/commit/e2c456147c260b4e9d583662e3bb9d6d9a019a5e))
- ✨ small edits ([bcf55ca](https://github.com/melMass/comfy_mtb/commit/bcf55ca9a3a07067be3319182501f7b635e5d2ba))
- ⚡️ add support for batch in roop ([2dae020](https://github.com/melMass/comfy_mtb/commit/2dae02056a11ddfe1f84ee040818028177e404b5))
- 🔥 various preparing for the first tag ([793784a](https://github.com/melMass/comfy_mtb/commit/793784a5fd08e8a70d670fc8edbc3bb5b6e13e67))
- 🐛 various bugs ([afd0843](https://github.com/melMass/comfy_mtb/commit/afd08431458e3bbb14a25c84a87408113edf5db5))
- ⚡️ add missing controls to QRCode ([7e86b0e](https://github.com/melMass/comfy_mtb/commit/7e86b0ed4d300021517f6c5cf28a45012497b5c5))

### Documentation

- 📝 add rembg screenshot ([9a2d523](https://github.com/melMass/comfy_mtb/commit/9a2d52325f87ecf6342ef4897da919006755b9db))
- 📝 add a few screenshots ([e162336](https://github.com/melMass/comfy_mtb/commit/e162336cd366d39cd4b96f05b3c9c68eecec3dc4))
- 📝 update readme ([7f3070d](https://github.com/melMass/comfy_mtb/commit/7f3070debbc3330da50ff845621ce299894cf862))

### Features

- 💄 faceswap node using roop ([966a14b](https://github.com/melMass/comfy_mtb/commit/966a14b40d88f4fccfb2eaa5ff9b222f0eedd7cb))
- ✨ sync local changes ([647bf9e](https://github.com/melMass/comfy_mtb/commit/647bf9e94195c279a620c74c2253471b9c4b90f7))
- ✨ bbox from alpha ([37abf8a](https://github.com/melMass/comfy_mtb/commit/37abf8aad12f4711c6d82c6be4be6fa3578e7af5))
- ✨ a111 like style loader ([f59b68e](https://github.com/melMass/comfy_mtb/commit/f59b68e3ad92841a4d189d8dddf7b41e915c9b4e))
- ✨ add a color type and widget ([9a2e986](https://github.com/melMass/comfy_mtb/commit/9a2e986327c34227a707beab6d9929b0a05e41e6))
- ✨ add a few nodes ([811443b](https://github.com/melMass/comfy_mtb/commit/811443b92161815db1cdff81898e8834dcd6fbfa))
- ✨ add SadTalker as a submodule ([3fb8716](https://github.com/melMass/comfy_mtb/commit/3fb871651b12bce62d8e911bd3884f417f80c937))
- 🚨 push local changes ([6cac344](https://github.com/melMass/comfy_mtb/commit/6cac344f6fb15ebb902acee70ee71edc585ec4bc))
- ⚡️ initial commit ([1ae3bbc](https://github.com/melMass/comfy_mtb/commit/1ae3bbc89ae6e0d2e8c61122485bd0df837e17c2))

### Miscellaneous Tasks

- 🚀 add gh action ([572b4d5](https://github.com/melMass/comfy_mtb/commit/572b4d52bce1398660d4d7ca0c5c48c11e0128e3)) in [#4](https://github.com/melMass/comfy_mtb/pull/4)

[main]: https://github.com/melMass/comfy_mtb/compare/v0.2.0..main
[0.2.0]: https://github.com/melMass/comfy_mtb/compare/v0.1.6..v0.2.0
[0.1.6]: https://github.com/melMass/comfy_mtb/compare/v0.1.5..v0.1.6
[0.1.5]: https://github.com/melMass/comfy_mtb/compare/v0.1.4..v0.1.5
[0.1.4]: https://github.com/melMass/comfy_mtb/compare/v0.1.3..v0.1.4
[0.1.3]: https://github.com/melMass/comfy_mtb/compare/v0.1.2..v0.1.3
[0.1.2]: https://github.com/melMass/comfy_mtb/compare/v0.1.1..v0.1.2
[0.1.1]: https://github.com/melMass/comfy_mtb/compare/v0.1.0..v0.1.1
[0.1.0]: https://github.com/melMass/comfy_mtb/compare/v0.0.1..v0.1.0

