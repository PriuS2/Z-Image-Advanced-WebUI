import { memo, useState, useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Upload, X, Image as ImageIcon, Loader2 } from 'lucide-react'
import { generationApi } from '../../api/generation'
import { useGenerationStore } from '../../stores/generationStore'
import { useToast } from '../../hooks/useToast'

function ImageInputNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { setParams } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  
  const [image, setImage] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const uploadFile = async (file: File) => {
    setIsUploading(true)
    try {
      // Show preview immediately
      const reader = new FileReader()
      reader.onload = (event) => {
        setImage(event.target?.result as string)
      }
      reader.readAsDataURL(file)

      // Upload to server
      const result = await generationApi.uploadImage(file)
      setParams({ controlImagePath: result.path })
      toastSuccess(t('common.success'), 'Image uploaded')
    } catch (err) {
      toastError(t('common.error'), String(err))
      setImage(null)
    } finally {
      setIsUploading(false)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      uploadFile(file)
    }
  }, [])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      uploadFile(file)
    }
  }, [])

  const clearImage = () => {
    setImage(null)
    setParams({ controlImagePath: undefined })
  }

  return (
    <div className={`min-w-[200px] rounded-lg border bg-card p-4 ${selected ? 'border-primary' : 'border-border'}`}>
      <div className="mb-3 text-sm font-semibold">{t('nodes.image')}</div>
      
      {image ? (
        <div className="relative">
          <img
            src={image}
            alt="Uploaded"
            className="w-full rounded-md object-cover max-h-40"
          />
          {isUploading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-md">
              <Loader2 className="h-6 w-6 animate-spin text-white" />
            </div>
          )}
          <button
            onClick={clearImage}
            disabled={isUploading}
            className="absolute -right-2 -top-2 rounded-full bg-destructive p-1 text-destructive-foreground disabled:opacity-50"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ) : (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="relative flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed
            border-muted-foreground/25 p-6 text-center hover:border-primary/50 transition-colors"
        >
          <Upload className="h-8 w-8 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            {t('generate.uploadImage')}
          </span>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="absolute inset-0 cursor-pointer opacity-0"
          />
        </div>
      )}
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="!bg-primary !w-3 !h-3"
      />
    </div>
  )
}

export const ImageInputNode = memo(ImageInputNodeComponent)
