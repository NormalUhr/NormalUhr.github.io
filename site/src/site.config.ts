export type Lang = 'en' | 'zh'

/**
 * Everything that is a personal choice rather than a design decision lives here.
 *
 * The two optional services are off until you fill them in. While they are null the
 * site simply does not render a read count or a comment section, so nothing on the
 * page is ever a made-up number.
 */
export const site = {
  name: { en: 'Yihua Zhang', zh: '张逸骅' },
  /** One line under the name on the home page, and the site's meta description.
   *  Left blank on purpose: nothing is shown until you write it. */
  tagline: { en: '', zh: '' } as Record<Lang, string>,
  /**
   * Path to an icon in site/public/, e.g. '/icon.svg'. Left blank because the mark is
   * yours to choose; while it is blank the page declares that it has no icon, which
   * stops browsers requesting a /favicon.ico that is not there.
   */
  favicon: null as string | null,
  since: 2019,
  links: [
    { label: 'GitHub', href: 'https://github.com/NormalUhr' },
    { label: 'Hugging Face', href: 'https://huggingface.co/NormalUhr' },
    { label: 'Google Scholar', href: 'https://scholar.google.com/citations?user=Xkc1MZoAAAAJ' },
    { label: 'X', href: 'https://x.com/zyh2022' },
    { label: '知乎', href: 'https://www.zhihu.com/people/chi-bo-li-de-xi' },
  ],

  /**
   * GoatCounter site code, i.e. the "yourname" in yourname.goatcounter.com.
   * Setting it turns on the page-view script, the per-article read count and the
   * site total in the footer. Leave null and none of that appears.
   */
  goatcounter: null as string | null,

  /**
   * Base URL of the likes Worker, e.g. 'https://blog-likes.yourname.workers.dev'.
   * See workers/likes/. Readers need no account; the count is per article and shared
   * between its languages. Leave null and no like button is rendered.
   */
  likes: 'https://blog-likes.zhangyihua1997.workers.dev' as string | null,

  /**
   * giscus, which keeps comment threads in a GitHub Discussions category.
   * Fill this in from https://giscus.app after enabling Discussions on the repo.
   * Leave null and no comment section is rendered.
   */
  giscus: null as null | {
    repo: string
    repoId: string
    category: string
    categoryId: string
  },
}

export const LANGS: Lang[] = ['en', 'zh']
export const DEFAULT_LANG: Lang = 'en'
