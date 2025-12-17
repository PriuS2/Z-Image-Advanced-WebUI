import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { generationApi } from '../../api/generation'

export function ProgressBar() {
  const { t } = useTranslation()
  const { progress, resetProgress } = useGenerationStore()

  if (!progress.isGenerating) {
    return null
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  const handleCancel = async () => {
    if (progress.taskId) {
      try {
        await generationApi.cancelTask(progress.taskId)
        resetProgress()
      } catch (error) {
        console.error('Failed to cancel task:', error)
      }
    }
  }

  return (
    <div className="border-b border-border bg-card/50 px-4 py-3">
      <div className="flex items-center justify-between gap-4">
        {/* Progress info */}
        <div className="flex items-center gap-4 flex-1">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
            <span className="text-sm font-medium">
              {t('common.loading')} ({progress.progress}%)
            </span>
          </div>
          
          <span className="text-sm text-muted-foreground">
            Step {progress.currentStep}/{progress.totalSteps}
          </span>
          
          {progress.elapsedTime > 0 && (
            <span className="text-sm text-muted-foreground">
              {formatTime(progress.elapsedTime)} / ~{formatTime(progress.elapsedTime + progress.estimatedRemaining)}
            </span>
          )}
          
          {progress.currentNode && (
            <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded">
              {t(`nodes.${progress.currentNode}`)}
            </span>
          )}
        </div>
        
        {/* Progress bar */}
        <div className="flex-1 max-w-md">
          <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
            <div 
              className="h-full bg-primary transition-all duration-300 ease-out"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
        </div>
        
        {/* Cancel button */}
        <button
          onClick={handleCancel}
          className="p-2 text-muted-foreground hover:text-destructive transition-colors"
          title={t('common.cancel')}
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
