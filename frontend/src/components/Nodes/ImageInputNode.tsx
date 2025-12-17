import { memo, useState, useCallback } from 'react'
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Upload, X, Image as ImageIcon, Loader2 } from 'lucide-react'
import { generationApi } from '../../api/generation'
import { useGenerationStore } from '../../stores/generationStore'
import { useToast } from '../../hooks/useToast'

function ImageInputNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { setParams } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  const { deleteElements } = useReactFlow()
  
  const [image, setImage] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

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
    <div 
      className={`min-w-[240px] rounded-lg border bg-card p-4 pr-14 relative ${selected ? 'border-primary' : 'border-border'}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Delete button */}
      {isHovered && (
        <button
          onClick={() => deleteElements({ nodes: [{ id }] })}
          className="absolute -right-2 -top-2 z-10 rounded-full bg-destructive p-1 text-destructive-foreground 
            hover:bg-destructive/80 transition-colors shadow-md"
        >
          <X className="h-3 w-3" />
        </button>
      )}
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
            className="nodrag absolute -right-2 -top-2 rounded-full bg-destructive p-1 text-destructive-foreground disabled:opacity-50"
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
            className="nodrag absolute inset-0 cursor-pointer opacity-0"
          />
        </div>
      )}
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="!bg-blue-500 !w-3 !h-3"
      />
      <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-blue-500 font-medium">
        Image
      </div>
    </div>
  )
}

export const ImageInputNode = memo(ImageInputNodeComponent)
