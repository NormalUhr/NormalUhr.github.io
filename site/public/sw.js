/*
 * A tombstone for the service worker the previous Jekyll theme registered at this path.
 * That worker raced its own cache against the network and answered with whichever
 * arrived first, so a reader's first view of a page they had seen before was always the
 * previous version and only a reload produced the current one. A registration survives
 * until the script at its path says otherwise, so this file takes its place, drops the
 * caches it left behind and unregisters itself. It can go once returning readers have
 * loaded the site once.
 */
self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) await caches.delete(name)
    await self.registration.unregister()
    for (const tab of await self.clients.matchAll({ type: 'window' })) {
      try {
        await tab.navigate(tab.url)
      } catch {
        // The tab may not be ours to navigate; the registration is gone either way.
      }
    }
  })())
})
