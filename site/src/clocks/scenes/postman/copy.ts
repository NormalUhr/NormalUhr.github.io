import type { Lang } from '../../../site.config'
import type { ClockText } from '../../scene'

export default {
  en: { name: 'The Time Postman' },
  zh: { name: '时间邮差' },
} satisfies Record<Lang, ClockText>
