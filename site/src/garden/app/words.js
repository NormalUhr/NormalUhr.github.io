/* Every word the garden shows, in Chinese, the original, and in English. The script picks the
   language from <html lang>, and the page reads WORDS for its fixed labels. */
export const WORDS = {
  zh: {
    title: '庭间 · 枯山水', close: '关闭',
    scene: '可以环绕观察和划沙的三维枯山水庭院。一家人从小屋轮流出来，整理划痕并维护时间螺旋。',
    light: '天光与季节', listen: '听听庭院', hush: '让庭院安静下来', clock: '看看时间',
    times: ['清晨', '白昼', '黄昏', '夜晚'], now: '回到此刻',
    weather: '天气', weathers: { auto: '随天气变化', clear: '晴朗', cloudy: '多云', mist: '薄雾', rain: '细雨', snow: '落雪' },
    season: '时节', seasonNow: '当下的季节', seasons: ['春', '夏', '秋', '冬'],
    views: '庭院观察', closer: '靠近看看', home: '回到全景', rake: '划沙',
    greetings: ['你那里现在几点啦？', '谢谢你愿意驻足停留。'],
    rakeHint: '在沙面上轻轻划过就好。',
    noSound: '这次没能听见声音，仍可以静静看看。',
    noScript: '请在浏览器中开启脚本，再来庭里坐坐。',
    failed: '这座小庭院暂时没能打开。',
    failedHelp: '请用较新的 Chrome、Edge 或 Safari 浏览器，并开启硬件加速。',
    retry: '再试一次',
  },
  en: {
    title: 'Karesansui Garden', close: 'Close',
    scene: 'A karesansui garden in 3D that you can look around and rake. A family takes turns coming out of the small house to smooth the marks and keep up the spiral of time.',
    light: 'Light and season', listen: 'Listen to the garden', hush: 'Let the garden be quiet', clock: 'Show the time',
    times: ['Morning', 'Daytime', 'Dusk', 'Night'], now: 'Back to now',
    weather: 'Weather', weathers: { auto: 'Changing', clear: 'Clear', cloudy: 'Cloudy', mist: 'Mist', rain: 'Drizzle', snow: 'Snow' },
    season: 'Season', seasonNow: 'Current', seasons: ['Spring', 'Summer', 'Autumn', 'Winter'],
    views: 'Garden views', closer: 'Up close', home: 'Full view', rake: 'Rake',
    greetings: ['What time is it where you are?', 'Thank you for staying a while.'],
    rakeHint: 'Draw lightly across the sand.',
    noSound: 'The garden could not play sound this time. You can still watch it quietly.',
    noScript: 'Turn on JavaScript in your browser, then come and sit in the garden.',
    failed: 'This little garden could not open just now.',
    failedHelp: 'Please use a recent Chrome, Edge or Safari, with hardware acceleration turned on.',
    retry: 'Try again',
  },
};

export const words = WORDS[globalThis.document?.documentElement.lang.startsWith('zh') ? 'zh' : 'en'];
