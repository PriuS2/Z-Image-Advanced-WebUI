import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Loader2 } from 'lucide-react'

const controlTypes = ['canny', 'hed', 'depth', 'pose', 'mlsd']

function ControlNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const [controlType, setControlType] = useState('canny')
  const [isExtracting, setIsExtracting] = useState(false)
  const [extractedImage, setExtractedImage] = useState<string | null>(null)

  const handleExtract = async () => {
    setIsExtracting(true)
    // TODO: Implement control extraction
    setTimeout(() => {
      setIsExtracting(false)
    }, 1000)
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
        disabled={isExtracting}
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
