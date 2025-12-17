import { memo, useState, useRef } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Loader2, Upload, X } from 'lucide-react'
import { generationApi } from '../../api/generation'
import { useGenerationStore } from '../../stores/generationStore'
import { useToast } from '../../hooks/useToast'

const controlTypes = ['canny', 'hed', 'depth', 'pose', 'mlsd']

function ControlNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { setParams } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  
  const [controlType, setControlType] = useState('canny')
  const [isExtracting, setIsExtracting] = useState(false)
  const [extractedImage, setExtractedImage] = useState<string | null>(null)
  const [sourceImage, setSourceImage] = useState<string | null>(null)
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSourceFile(file)
      const reader = new FileReader()
      reader.onload = (event) => {
        setSourceImage(event.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleExtract = async () => {
    if (!sourceFile) {
      toastError(t('common.error'), 'Please upload an image first')
      return
    }

    setIsExtracting(true)
    try {
      const result = await generationApi.extractControl(sourceFile, controlType)
      setExtractedImage(result.control_image_path)
      setParams({ controlType, controlImagePath: result.control_image_path })
      toastSuccess(t('common.success'), 'Control image extracted')
    } catch (err) {
      toastError(t('common.error'), String(err))
    } finally {
      setIsExtracting(false)
    }
  }

  const clearImage = () => {
    setSourceImage(null)
    setSourceFile(null)
    setExtractedImage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className={`min-w-[200px] rounded-lg border bg-card p-4 ${selected ? 'border-primary' : 'border-border'}`}>
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="!bg-primary !w-3 !h-3"
      />
      
      <div className="mb-3 text-sm font-semibold">{t('nodes.control')}</div>
      
      {/* Source image upload */}
      <div className="mb-3">
        {sourceImage ? (
          <div className="relative">
            <img
              src={sourceImage}
              alt="Source"
              className="w-full rounded-md object-cover max-h-24"
            />
            <button
              onClick={clearImage}
              className="absolute -right-2 -top-2 rounded-full bg-destructive p-1 text-destructive-foreground"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ) : (
          <div className="relative">
            <div className="flex flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed
              border-muted-foreground/25 p-3 text-center hover:border-primary/50 transition-colors cursor-pointer">
              <Upload className="h-5 w-5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">{t('generate.uploadImage')}</span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="absolute inset-0 cursor-pointer opacity-0"
            />
          </div>
        )}
      </div>
      
      {/* Control type selector */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.controlType')}
        </label>
        <select
          value={controlType}
          onChange={(e) => setControlType(e.target.value)}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm
            focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {controlTypes.map((type) => (
            <option key={type} value={type}>
              {type.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
      
      {/* Extract button */}
      <button
        onClick={handleExtract}
        disabled={isExtracting || !sourceImage}
        className="w-full flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm
          bg-secondary text-secondary-foreground hover:bg-secondary/80
          disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isExtracting && <Loader2 className="h-4 w-4 animate-spin" />}
        {t('generate.extractControl')}
      </button>
      
      {/* Preview of extracted control */}
      {extractedImage && (
        <div className="mt-3">
          <label className="mb-1 block text-xs text-muted-foreground">Extracted</label>
          <img
            src={extractedImage}
            alt="Control"
            className="w-full rounded-md object-cover"
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

export const ControlNode = memo(ControlNodeComponent)
