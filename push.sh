#!/usr/bin/env bash
# 一键将本地代码提交并推送到 GitHub
# 用法:
#   ./push.sh              # 使用默认提交信息
#   ./push.sh "更新说明"   # 自定义提交信息
#   ./push.sh -m "说明"    # 同上
#   ./push.sh --dry-run    # 仅预览，不提交/推送

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
DRY_RUN=0
COMMIT_MSG=""

usage() {
  cat <<'EOF'
用法: ./push.sh [选项] [提交信息]

选项:
  -m, --message <msg>   提交说明（默认: update: YYYY-MM-DD HH:MM:SS）
  -r, --remote <name>   远程名（默认: origin，可用环境变量 GIT_REMOTE）
  -b, --branch <name>   推送分支（默认: 当前分支）
  -n, --dry-run         只显示将要做什么，不执行
  -h, --help            显示帮助

示例:
  ./push.sh
  ./push.sh "feat: 添加自动标注流程"
  ./push.sh -m "fix: 修复数据加载" -b main
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      COMMIT_MSG="${2:-}"
      shift 2
      ;;
    -r|--remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    -b|--branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "未知选项: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      COMMIT_MSG="$1"
      shift
      ;;
  esac
done

if [[ -z "$COMMIT_MSG" ]]; then
  COMMIT_MSG="update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误: 当前目录不是 git 仓库" >&2
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "错误: 未配置远程仓库 '$REMOTE'" >&2
  echo "请先执行: git remote add origin <你的 GitHub 仓库 URL>" >&2
  exit 1
fi

REMOTE_URL="$(git remote get-url "$REMOTE")"
echo "仓库: $ROOT_DIR"
echo "远程: $REMOTE ($REMOTE_URL)"
echo "分支: $BRANCH"
echo "说明: $COMMIT_MSG"
echo

# 显示变更概览
if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
  # 本地无未提交变更，检查是否需要推送已有提交
  if git rev-parse --verify "@{u}" >/dev/null 2>&1; then
    AHEAD="$(git rev-list --count "@{u}..HEAD" 2>/dev/null || echo 0)"
  else
    AHEAD=1
  fi

  if [[ "$AHEAD" -eq 0 ]]; then
    echo "没有需要提交或推送的变更，已与远程同步。"
    exit 0
  fi

  echo "本地无新文件变更，但有 $AHEAD 个提交尚未推送。"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] 将执行: git push -u $REMOTE $BRANCH"
    exit 0
  fi
  git push -u "$REMOTE" "$BRANCH"
  echo
  echo "推送完成: $REMOTE_URL"
  exit 0
fi

echo "变更预览:"
git status -sb
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] 将执行:"
  echo "  git add -A"
  echo "  git commit -m \"$COMMIT_MSG\""
  echo "  git push -u $REMOTE $BRANCH"
  exit 0
fi

git add -A

# 再次确认暂存区是否有内容（避免空提交）
if git diff --cached --quiet; then
  echo "暂存区为空（可能全部被 .gitignore 忽略），跳过提交。"
else
  git commit -m "$COMMIT_MSG"
fi

git push -u "$REMOTE" "$BRANCH"

echo
echo "上传完成 → $REMOTE_URL ($BRANCH)"
