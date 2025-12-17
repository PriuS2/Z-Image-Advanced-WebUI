import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Zap, Loader2 } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'

export function LoginDialog() {
  const { t } = useTranslation()
  const { login, register, isLoading, error, clearError } = useAuthStore()
  
  const [isRegisterMode, setIsRegisterMode] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (isRegisterMode) {
      await register(username, password)
    } else {
      await login(username, password)
    }
  }

  const toggleMode = () => {
    setIsRegisterMode(!isRegisterMode)
    clearError()
  }

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-8 shadow-lg">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center gap-2">
          <Zap className="h-12 w-12 text-primary" />
          <h1 className="text-2xl font-bold">{t('app.title')}</h1>
          <p className="text-sm text-muted-foreground">{t('app.subtitle')}</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="username" className="text-sm font-medium">
              {t('auth.username')}
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="rounded-md border border-input bg-background px-3 py-2 text-sm
                ring-offset-background placeholder:text-muted-foreground
                focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              placeholder={t('auth.username')}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="password" className="text-sm font-medium">
              {t('auth.password')}
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-md border border-input bg-background px-3 py-2 text-sm
                ring-offset-background placeholder:text-muted-foreground
                focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              placeholder={t('auth.password')}
              required
            />
          </div>

          {error && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2
              text-sm font-medium text-primary-foreground
              hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors"
          >
            {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            {isRegisterMode ? t('auth.registerButton') : t('auth.loginButton')}
          </button>
        </form>

        {/* Toggle mode */}
        <div className="mt-6 text-center text-sm text-muted-foreground">
          {isRegisterMode ? t('auth.hasAccount') : t('auth.noAccount')}{' '}
          <button
            type="button"
            onClick={toggleMode}
            className="text-primary hover:underline"
          >
            {isRegisterMode ? t('auth.login') : t('auth.register')}
          </button>
        </div>
      </div>
    </div>
  )
}
