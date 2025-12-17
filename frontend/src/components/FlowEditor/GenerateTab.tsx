import { useCallback, useState } from 'react'
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

const initialEdges: Edge[] = [
  { id: 'e1-3', source: 'prompt-1', target: 'generate-1', sourceHandle: 'output', targetHandle: 'prompt' },
  { id: 'e2-3', source: 'parameters-1', target: 'generate-1', sourceHandle: 'output', targetHandle: 'params' },
  { id: 'e3-4', source: 'generate-1', target: 'preview-1', sourceHandle: 'output', targetHandle: 'input' },
]

export function GenerateTab() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
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
      if (!type) return

      const position = {
        x: event.clientX - 250,
        y: event.clientY - 100,
      }

      const newNode: Node = {
        id: `${type}-${nodes.length + 1}`,
        type,
        position,
        data: { label: type },
      }

      setNodes((nds) => nds.concat(newNode))
    },
    [nodes, setNodes]
  )

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
          nodeTypes={nodeTypes}
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
