import { useCallback, useState, useRef } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  BackgroundVariant,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  MarkerType,
  ReactFlowInstance,
} from 'reactflow'
import 'reactflow/dist/style.css'

import { PromptNode } from '../Nodes/PromptNode'
import { ImageInputNode } from '../Nodes/ImageInputNode'
import { ControlNode } from '../Nodes/ControlNode'
import { MaskNode } from '../Nodes/MaskNode'
import { ParametersNode } from '../Nodes/ParametersNode'
import { GenerateNode } from '../Nodes/GenerateNode'
import { PreviewNode } from '../Nodes/PreviewNode'

const nodeTypes = {
  prompt: PromptNode,
  imageInput: ImageInputNode,
  control: ControlNode,
  mask: MaskNode,
  parameters: ParametersNode,
  generate: GenerateNode,
  preview: PreviewNode,
}

// 핸들 타입 정의 (노드타입:핸들ID -> 데이터타입)
const handleDataTypes: Record<string, Record<string, string>> = {
  // 출력 핸들 (source)
  prompt: { output: 'prompt' },
  parameters: { output: 'params' },
  imageInput: { output: 'image' },
  control: { input: 'image', output: 'control' },
  mask: { input: 'image', output: 'mask' },
  generate: { 
    prompt: 'prompt', 
    params: 'params', 
    image: 'image',    // ImageInputNode 연결용
    control: 'control', 
    mask: 'mask',      // MaskNode 연결용
    output: 'image' 
  },
  preview: { input: 'image' },
}

// 연결 가능 여부 확인
const getHandleDataType = (nodeType: string | undefined, handleId: string | null): string | null => {
  if (!nodeType || !handleId) return null
  return handleDataTypes[nodeType]?.[handleId] || null
}

const initialNodes: Node[] = [
  {
    id: 'prompt-1',
    type: 'prompt',
    position: { x: 50, y: 100 },
    data: { label: 'Prompt' },
  },
  {
    id: 'parameters-1',
    type: 'parameters',
    position: { x: 50, y: 350 },
    data: { label: 'Parameters' },
  },
  {
    id: 'generate-1',
    type: 'generate',
    position: { x: 400, y: 200 },
    data: { label: 'Generate' },
  },
  {
    id: 'preview-1',
    type: 'preview',
    position: { x: 700, y: 200 },
    data: { label: 'Preview' },
  },
]

// 타입별 엣지 색상
const typeColors: Record<string, string> = {
  prompt: '#22c55e',   // 녹색
  params: '#a855f7',   // 보라색
  image: '#3b82f6',    // 파란색
  control: '#f97316',  // 주황색
  mask: '#ec4899',     // 핑크색
}

const createEdgeStyle = (type: string) => ({
  stroke: typeColors[type] || '#888',
  strokeWidth: 2,
})

const initialEdges: Edge[] = [
  { 
    id: 'e1-3', 
    source: 'prompt-1', 
    target: 'generate-1', 
    sourceHandle: 'output', 
    targetHandle: 'prompt',
    style: createEdgeStyle('prompt'),
    markerEnd: { type: MarkerType.ArrowClosed, color: typeColors.prompt },
  },
  { 
    id: 'e2-3', 
    source: 'parameters-1', 
    target: 'generate-1', 
    sourceHandle: 'output', 
    targetHandle: 'params',
    style: createEdgeStyle('params'),
    markerEnd: { type: MarkerType.ArrowClosed, color: typeColors.params },
  },
  { 
    id: 'e3-4', 
    source: 'generate-1', 
    target: 'preview-1', 
    sourceHandle: 'output', 
    targetHandle: 'input',
    style: createEdgeStyle('image'),
    markerEnd: { type: MarkerType.ArrowClosed, color: typeColors.image },
  },
]

export function GenerateTab() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null)

  // 연결 유효성 검사 - 같은 타입끼리만 연결 가능
  const isValidConnection = useCallback(
    (connection: Connection) => {
      const sourceNode = nodes.find((n) => n.id === connection.source)
      const targetNode = nodes.find((n) => n.id === connection.target)

      const sourceType = getHandleDataType(sourceNode?.type, connection.sourceHandle)
      const targetType = getHandleDataType(targetNode?.type, connection.targetHandle)

      // 타입이 같아야 연결 가능
      return sourceType !== null && targetType !== null && sourceType === targetType
    },
    [nodes]
  )

  const onConnect = useCallback(
    (connection: Connection) => {
      const sourceNode = nodes.find((n) => n.id === connection.source)
      const dataType = getHandleDataType(sourceNode?.type, connection.sourceHandle)
      const color = typeColors[dataType || ''] || '#888'

      const newEdge = {
        ...connection,
        id: `e-${connection.source}-${connection.target}-${Date.now()}`,
        style: { stroke: color, strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color },
      }

      setEdges((eds) => addEdge(newEdge, eds))
    },
    [setEdges, nodes]
  )

  // 엣지 삭제 (마우스 우클릭)
  const onEdgeContextMenu = useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      event.preventDefault()
      setEdges((eds) => eds.filter((e) => e.id !== edge.id))
    },
    [setEdges]
  )

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const type = event.dataTransfer.getData('application/reactflow')
      if (!type || !reactFlowInstance.current) return

      // 스크린 좌표를 Flow 좌표로 변환
      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      })

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { label: type },
      }

      setNodes((nds) => nds.concat(newNode))
    },
    [setNodes]
  )

  const onInit = useCallback((instance: ReactFlowInstance) => {
    reactFlowInstance.current = instance
  }, [])

  return (
    <div className="flex h-full gap-4">
      {/* Node Palette */}
      <div className="w-48 flex-shrink-0 rounded-lg border border-border bg-card p-4">
        <h3 className="mb-4 font-semibold text-sm">Nodes</h3>
        <div className="flex flex-col gap-2">
          {Object.keys(nodeTypes).map((type) => (
            <div
              key={type}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('application/reactflow', type)
                e.dataTransfer.effectAllowed = 'move'
              }}
              className="cursor-grab rounded-md border border-border bg-background px-3 py-2
                text-sm capitalize hover:border-primary hover:bg-accent transition-colors"
            >
              {type}
            </div>
          ))}
        </div>
      </div>

      {/* Flow Editor */}
      <div className="flex-1 rounded-lg border border-border overflow-hidden">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onInit={onInit}
          onEdgeContextMenu={onEdgeContextMenu}
          isValidConnection={isValidConnection}
          nodeTypes={nodeTypes}
          deleteKeyCode={['Backspace', 'Delete']}
          fitView
          className="bg-background"
        >
          <Controls className="bg-card border-border" />
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        </ReactFlow>
      </div>
    </div>
  )
}
