import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Plus, Save, Trash2, Edit, Play, Loader2, Workflow } from 'lucide-react'
import { apiClient, handleApiError } from '../../api/client'
import { useToast } from '../../hooks/useToast'

interface WorkflowItem {
  id: number
  name: string
  description?: string
  thumbnail_path?: string
  created_at: string
  updated_at: string
  is_favorite: boolean
}

export function WorkflowTab() {
  const { t } = useTranslation()
  const { error: toastError, success: toastSuccess } = useToast()
  
  const [workflows, setWorkflows] = useState<WorkflowItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [newWorkflowName, setNewWorkflowName] = useState('')
  const [newWorkflowDesc, setNewWorkflowDesc] = useState('')

  const loadWorkflows = async () => {
    try {
      const response = await apiClient.get<WorkflowItem[]>('/workflow/')
      setWorkflows(response.data)
    } catch (err) {
      // Endpoint might not exist yet
      console.log('Workflow API not available yet')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadWorkflows()
  }, [])

  const handleCreate = async () => {
    if (!newWorkflowName.trim()) return

    try {
      const response = await apiClient.post('/workflow/', {
        name: newWorkflowName,
        description: newWorkflowDesc,
        nodes_data: { nodes: [], edges: [] },
      })
      setWorkflows((prev) => [response.data, ...prev])
      setShowNewDialog(false)
      setNewWorkflowName('')
      setNewWorkflowDesc('')
      toastSuccess(t('common.success'), 'Workflow created')
    } catch (err) {
      toastError(t('errors.networkError'), handleApiError(err))
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await apiClient.delete(`/workflow/${id}`)
      setWorkflows((prev) => prev.filter((w) => w.id !== id))
    } catch (err) {
      toastError(t('errors.networkError'), handleApiError(err))
    }
  }

  const handleLoad = async (id: number) => {
    try {
      const response = await apiClient.get(`/workflow/${id}`)
      // TODO: Load workflow into flow editor
      toastSuccess(t('common.success'), 'Workflow loaded')
    } catch (err) {
      toastError(t('errors.networkError'), handleApiError(err))
    }
  }

  return (
    <div className="h-full">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold">{t('workflow.title')}</h2>
        
        <button
          onClick={() => setShowNewDialog(true)}
          className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm 
            text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <Plus className="h-4 w-4" />
          {t('workflow.new')}
        </button>
      </div>

      {/* New workflow dialog */}
      {showNewDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6">
            <h3 className="mb-4 text-lg font-semibold">{t('workflow.new')}</h3>
            
            <div className="mb-4">
              <label className="mb-1 block text-sm font-medium">{t('workflow.name')}</label>
              <input
                type="text"
                value={newWorkflowName}
                onChange={(e) => setNewWorkflowName(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="My Workflow"
              />
            </div>
            
            <div className="mb-4">
              <label className="mb-1 block text-sm font-medium">{t('workflow.description')}</label>
              <textarea
                value={newWorkflowDesc}
                onChange={(e) => setNewWorkflowDesc(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm resize-none"
                rows={3}
                placeholder="Description..."
              />
            </div>
            
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowNewDialog(false)}
                className="rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleCreate}
                disabled={!newWorkflowName.trim()}
                className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground 
                  hover:bg-primary/90 disabled:opacity-50"
              >
                {t('common.save')}
              </button>
            </div>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : workflows.length === 0 ? (
        <div className="flex h-64 flex-col items-center justify-center gap-4 text-muted-foreground">
          <Workflow className="h-16 w-16" />
          <p>{t('workflow.noWorkflows')}</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {workflows.map((workflow) => (
            <div
              key={workflow.id}
              className="group rounded-lg border border-border bg-card p-4 hover:border-primary/50 transition-colors"
            >
              {/* Thumbnail */}
              <div className="mb-3 aspect-video rounded-md bg-muted/50 flex items-center justify-center">
                {workflow.thumbnail_path ? (
                  <img
                    src={workflow.thumbnail_path}
                    alt={workflow.name}
                    className="h-full w-full object-cover rounded-md"
                  />
                ) : (
                  <Workflow className="h-12 w-12 text-muted-foreground" />
                )}
              </div>
              
              {/* Info */}
              <h3 className="font-medium truncate">{workflow.name}</h3>
              {workflow.description && (
                <p className="text-sm text-muted-foreground truncate">{workflow.description}</p>
              )}
              <p className="text-xs text-muted-foreground mt-1">
                {new Date(workflow.updated_at).toLocaleDateString()}
              </p>
              
              {/* Actions */}
              <div className="mt-3 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => handleLoad(workflow.id)}
                  className="flex-1 flex items-center justify-center gap-1 rounded-md bg-primary py-1.5 text-sm
                    text-primary-foreground hover:bg-primary/90"
                >
                  <Play className="h-3 w-3" />
                  {t('workflow.load')}
                </button>
                <button
                  onClick={() => handleDelete(workflow.id)}
                  className="rounded-md bg-secondary p-1.5 hover:bg-destructive/10 hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
