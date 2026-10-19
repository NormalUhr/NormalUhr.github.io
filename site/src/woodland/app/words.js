/* Every word the woodland shows, in Chinese, the original, and in English. The script picks the
   language from <html lang>, and the page reads WORDS for its fixed labels. */
export const WORDS = {
  zh: {
    title: '林间时钟', close: '关闭',
    scene: '三维蚂蚁林地时钟。拖拽环绕观察，滚轮或双指缩放，右键或长按地面落石。',
    now: '此刻的林地', time: '当前时间', climate: '林地环境', place: '林地', views: '观察视角',
    closer: '靠近蚁群', closerLabel: '靠近蚁群观察',
    reset: '回到全景', resetLabel: '回到林地全景',
    quiet: '沉浸观察', quietLabel: '切换沉浸观察', show: '显示界面',
    greetings: ['你那里现在几点啦？', '谢谢你愿意驻足停留。'],
    seasons: ['春', '夏', '秋', '冬'],
    weather: { clear: '晴朗', cloudy: '多云', rain: '细雨', storm: '阵雨', snow: '落雪', mist: '薄雾' },
    degrees: (n) => n + '°',
    failed: '林地暂时没有打开。',
    failedHelp: '请刷新重试，或在浏览器设置中开启硬件加速后再来。',
  },
  en: {
    title: 'Woodland Clock', close: 'Close',
    scene: 'A woodland clock in 3D, kept by ants. Drag to look around, scroll or pinch to zoom, and right-click or long-press the ground to drop a stone.',
    now: 'The woodland now', time: 'Current time', climate: 'Weather in the woodland', place: 'Woodland', views: 'Views',
    closer: 'Up close', closerLabel: 'Watch the colony up close',
    reset: 'Full view', resetLabel: 'Back to the full view of the woodland',
    quiet: 'Immersive', quietLabel: 'Immersive view', show: 'Show controls',
    greetings: ['What time is it where you are?', 'Thank you for staying a while.'],
    seasons: ['Spring', 'Summer', 'Autumn', 'Winter'],
    weather: { clear: 'Clear', cloudy: 'Cloudy', rain: 'Drizzle', storm: 'Showers', snow: 'Snow', mist: 'Mist' },
    degrees: (n) => n + '°C',
    failed: 'The woodland could not open just now.',
    failedHelp: 'Reload to try again, or turn on hardware acceleration in your browser settings and come back.',
  },
};

export const words = WORDS[globalThis.document?.documentElement.lang.startsWith('zh') ? 'zh' : 'en'];
