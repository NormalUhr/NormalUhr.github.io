/**
 * Secrets are not part of wrangler.jsonc, so `wrangler types` cannot see them.
 * Declaration-merge them into the generated global Env here.
 *
 *   wrangler secret put VOTE_SALT
 */
interface Env {
  /** Any long random string. Salts the reader hash so addresses are not recoverable. */
  VOTE_SALT: string
}
