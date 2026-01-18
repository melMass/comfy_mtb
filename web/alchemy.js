import { app } from "../../scripts/app.js";

  app.registerExtension({
    name: "PromptAlchemy.Bridge",

    setup() {
      // Listen for messages from parent window
      window.addEventListener('message', async (event) => {
        // Optionally verify origin
        const { type, ...payload } = event.data || {};

        switch (type) {
          case 'loadWorkflow':
            await app.loadGraphData(payload.workflow);
            this.sendToParent({ type: 'workflowLoaded' });
            break;

          case 'getWorkflow':
            const workflow = app.graph.serialize();
            this.sendToParent({ type: 'workflow', workflow });
            break;

          case 'queuePrompt':
            app.queuePrompt(0); // 0 = front of queue
            break;

          case 'getPrompt':
            const prompt = await app.graphToPrompt();
            this.sendToParent({ type: 'prompt', prompt });
            break;
        }
      });
	
	  console.log("Alchemy Bridge Extension Loaded");

      // Notify parent we're ready
      this.sendToParent({ type: 'ready' });
    },

    sendToParent(data) {
      if (window.parent !== window) {
        window.parent.postMessage(data, '*');
      }
    }
  });


