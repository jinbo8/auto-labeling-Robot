import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";

type Ann = {
  id?: string;
  frame_index: number;
  camera_key: string;
  label: string;
  geometry: { type: string; bbox?: number[] };
  source: string;
  track_id?: string | null;
};

export default function StudioPage() {
  const { jobId } = useParams();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [job, setJob] = useState<any>(null);
  const [ontology, setOntology] = useState<any>(null);
  const [label, setLabel] = useState("cube");
  const [camera, setCamera] = useState("");
  const [frame, setFrame] = useState(0);
  const [anns, setAnns] = useState<Ann[]>([]);
  const [preds, setPreds] = useState<any[]>([]);
  const [drawing, setDrawing] = useState<number[] | null>(null);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const cameras = useMemo(() => Object.keys(job?.episode?.video_refs || {}), [job]);

  const maxFrame = (job?.episode?.length || 1) - 1;

  async function load() {
    const j = await api<any>(`/api/v1/jobs/${jobId}`);
    setJob(j);
    const cams = Object.keys(j.episode?.video_refs || {});
    setCamera((c) => c || cams[0] || "");
    const ont = await api<any>(`/api/v1/projects/${j.project_id}/ontology`);
    setOntology(ont);
    if (ont.labels?.[0]?.name) setLabel(ont.labels[0].name);
    const a = await api<Ann[]>(`/api/v1/jobs/${jobId}/annotations`);
    setAnns(a);
    setPreds(await api(`/api/v1/jobs/${jobId}/predictions`));
  }

  useEffect(() => {
    load().catch((e) => setErr(String(e)));
  }, [jobId]);

  useEffect(() => {
    draw();
  }, [anns, preds, frame, camera, drawing]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#0a0e13";
    ctx.fillRect(0, 0, w, h);
    // placeholder frame grid
    ctx.strokeStyle = "#1e2a36";
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.fillStyle = "#8b9aab";
    ctx.font = "14px sans-serif";
    ctx.fillText(`${camera || "camera"} · frame ${frame}`, 16, 24);
    ctx.fillText("MVP 画布：矩形标注（视频帧预览轨后续接入）", 16, 44);

    const scaleX = w / 640;
    const scaleY = h / 480;
    for (const p of preds.filter((x) => x.frame_index === frame && x.camera_key === camera)) {
      const b = p.geometry?.bbox;
      if (!b) continue;
      ctx.strokeStyle = "rgba(240,180,41,0.9)";
      ctx.lineWidth = 2;
      ctx.strokeRect(b[0] * scaleX, b[1] * scaleY, b[2] * scaleX, b[3] * scaleY);
      ctx.fillStyle = "rgba(240,180,41,0.85)";
      ctx.fillText(`pred:${p.label}`, b[0] * scaleX, b[1] * scaleY - 4);
    }
    for (const a of anns.filter((x) => x.frame_index === frame && x.camera_key === camera)) {
      const b = a.geometry?.bbox;
      if (!b) continue;
      ctx.strokeStyle = "#3ecf8e";
      ctx.lineWidth = 2;
      ctx.strokeRect(b[0] * scaleX, b[1] * scaleY, b[2] * scaleX, b[3] * scaleY);
      ctx.fillStyle = "#3ecf8e";
      ctx.fillText(a.label, b[0] * scaleX, b[1] * scaleY - 4);
    }
    if (drawing) {
      const [x0, y0, x1, y1] = drawing;
      ctx.strokeStyle = "#3d9cf0";
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
  }

  function toImageXY(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  }

  function onDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const { x, y } = toImageXY(e);
    setDrawing([x, y, x, y]);
  }
  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawing) return;
    const { x, y } = toImageXY(e);
    setDrawing([drawing[0], drawing[1], x, y]);
  }
  function onUp() {
    if (!drawing || !camera) return;
    const canvas = canvasRef.current!;
    const scaleX = 640 / canvas.width;
    const scaleY = 480 / canvas.height;
    const x0 = Math.min(drawing[0], drawing[2]) * scaleX;
    const y0 = Math.min(drawing[1], drawing[3]) * scaleY;
    const x1 = Math.max(drawing[0], drawing[2]) * scaleX;
    const y1 = Math.max(drawing[1], drawing[3]) * scaleY;
    const bbox = [x0, y0, x1 - x0, y1 - y0];
    if (bbox[2] < 4 || bbox[3] < 4) {
      setDrawing(null);
      return;
    }
    setAnns((prev) => [
      ...prev,
      {
        frame_index: frame,
        camera_key: camera,
        label,
        geometry: { type: "bbox", bbox },
        source: "human",
      },
    ]);
    setDrawing(null);
  }

  async function save() {
    setErr("");
    try {
      const saved = await api<Ann[]>(`/api/v1/jobs/${jobId}/annotations`, {
        method: "PUT",
        body: JSON.stringify({ items: anns }),
      });
      setAnns(saved);
      setMsg("已保存标注");
    } catch (e) {
      setErr(String(e));
    }
  }

  async function acceptPreds() {
    await api(`/api/v1/jobs/${jobId}/predictions/accept`, {
      method: "POST",
      body: JSON.stringify({ min_score: 0.5 }),
    });
    await load();
    setMsg("已接受预标");
  }

  async function sam3() {
    setErr("");
    try {
      const r = await api<any>("/api/v1/models/sam3/predict", {
        method: "POST",
        body: JSON.stringify({ image_id: `${jobId}:${camera}:${frame}`, text: label }),
      });
      const box = r.masks?.[0]?.bbox || (r.boxes?.[0] && [
        r.boxes[0][0],
        r.boxes[0][1],
        r.boxes[0][2] - r.boxes[0][0],
        r.boxes[0][3] - r.boxes[0][1],
      ]);
      if (box) {
        setAnns((prev) => [
          ...prev,
          {
            frame_index: frame,
            camera_key: camera,
            label,
            geometry: { type: "bbox", bbox: box },
            source: "sam3",
          },
        ]);
        setMsg(`SAM3 stub 命中 score=${r.scores?.[0]}`);
      }
    } catch (e) {
      setErr(String(e));
    }
  }

  async function submit() {
    await save();
    await api(`/api/v1/jobs/${jobId}/submit`, { method: "POST" });
    setMsg("已提交审核");
    await load();
  }

  async function review(decision: "accept" | "reject") {
    await api(`/api/v1/jobs/${jobId}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, issues: [] }),
    });
    setMsg(`审核 ${decision}`);
    await load();
  }

  return (
    <div>
      <div className="row" style={{ marginBottom: "0.75rem" }}>
        <Link to={job ? `/projects/${job.project_id}` : "/projects"}>← 返回项目</Link>
        <span className="muted">Job {jobId?.slice(0, 8)} · ep {job?.episode_index} · {job?.status}</span>
        {msg && <span className="muted">{msg}</span>}
        {err && <span className="error">{err}</span>}
      </div>
      <div className="studio">
        <div className="card" style={{ margin: 0, overflow: "auto" }}>
          <h3>资源</h3>
          <p className="muted">{job?.episode?.task_text}</p>
          <label className="muted">相机</label>
          <select value={camera} onChange={(e) => setCamera(e.target.value)} style={{ width: "100%" }}>
            {cameras.map((c: string) => (
              <option key={c} value={c}>{c.split(".").pop()}</option>
            ))}
          </select>
        </div>
        <div className="canvas-wrap">
          <canvas
            ref={canvasRef}
            width={960}
            height={540}
            onMouseDown={onDown}
            onMouseMove={onMove}
            onMouseUp={onUp}
          />
        </div>
        <div className="card" style={{ margin: 0, overflow: "auto" }}>
          <h3>标签 / 模型</h3>
          <select value={label} onChange={(e) => setLabel(e.target.value)} style={{ width: "100%" }}>
            {(ontology?.labels || [{ name: "cube" }]).map((l: any) => (
              <option key={l.name} value={l.name}>{l.name}</option>
            ))}
          </select>
          <div className="row" style={{ marginTop: "0.75rem" }}>
            <button type="button" onClick={sam3}>SAM3 当前帧</button>
            <button type="button" className="secondary" onClick={acceptPreds}>接受预标</button>
          </div>
          <div className="row" style={{ marginTop: "0.75rem" }}>
            <button type="button" onClick={save}>保存</button>
            <button type="button" className="secondary" onClick={submit}>提交</button>
          </div>
          <div className="row" style={{ marginTop: "0.75rem" }}>
            <button type="button" className="secondary" onClick={() => review("accept")}>审核通过</button>
            <button type="button" className="secondary" onClick={() => review("reject")}>驳回</button>
          </div>
          <p className="muted" style={{ marginTop: "1rem" }}>
            预标 {preds.length} · 标注 {anns.length}
          </p>
        </div>
        <div className="timeline row">
          <button type="button" className="secondary" onClick={() => setFrame((f) => Math.max(0, f - 1))}>←</button>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxFrame)}
            value={frame}
            onChange={(e) => setFrame(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <button type="button" className="secondary" onClick={() => setFrame((f) => Math.min(maxFrame, f + 1))}>→</button>
          <input
            type="number"
            value={frame}
            min={0}
            max={maxFrame}
            onChange={(e) => setFrame(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </div>
      </div>
    </div>
  );
}
