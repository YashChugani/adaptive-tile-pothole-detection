"""End-to-end inference pipeline (stub).

Wires together the two-stage, decoupled design (see CLAUDE.md):

    image/frame
      -> adaptive tiling controller   (src.tiling)   decide whether/how to tile
      -> YOLO11 detector              (src.detection) full image + tiles
      -> cross-tile NMS merge
      -> learned severity head        (src.severity)  score each detected crop
      -> boxes + severity + summary

Nothing is implemented yet. Build components one at a time, in order:
dataset prep -> baseline detector -> adaptive tiling -> severity head ->
this pipeline + demo.
"""


def main() -> None:
    raise NotImplementedError("Pipeline not implemented yet. See CLAUDE.md build order.")


if __name__ == "__main__":
    main()
