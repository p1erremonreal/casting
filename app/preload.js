const { contextBridge, ipcRenderer } = require('electron');

const on = (channel) => (cb) => ipcRenderer.on(channel, (_e, payload) => cb(payload));

contextBridge.exposeInMainWorld('bridge', {
  onFrame: on('frame'),
  onScene: on('scene'),
  onStatus: on('status'),
});
