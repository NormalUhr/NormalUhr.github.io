import type { Lang } from '../../../site.config'

export default {
  en: {
    sign: 'Street {n}',
    reading: 'The postman is delivering to Number {number} on Street {street}',
    months: 'JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC',
  },
  zh: {
    sign: '{n} 街',
    reading: '邮差正在给 {street} 街 {number} 号送信',
    months: '1月 2月 3月 4月 5月 6月 7月 8月 9月 10月 11月 12月',
  },
} satisfies Record<Lang, Record<string, string>>
