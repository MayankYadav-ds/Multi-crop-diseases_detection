# Multi-Crop Plant Disease Detection and Severity Estimation (Offline)

University AI/ML course project: one leaf photo → crop–disease label and severity proxy (mild, moderate, severe). A shared MobileNetV3-Small backbone has two classification heads. Masks supply severity targets during development; inference requires only the photograph, exported TorchScript model, JSON and installed runtime dependencies.

## Run in Google Colab

1. Open `Milestone2_Demo_Notebook.ipynb` in Colab. Select **Runtime → Change runtime type → GPU** (e.g. T4, subject to availability).
2. Run sections 1–3, verify the detected metadata columns, split values and file matching. The notebook downloads `Plantseg.zip` from Zenodo record 16665770, version 6. If the download fails, remove any incomplete archive, upload the original ZIP to Drive, mount Drive as shown in section 2, and set `ZIP_PATH` accordingly. Set `META_CSV` or `COLUMN_OVERRIDES` in section 1 if needed; rerun from setup.
3. Select **Runtime → Run all** after verifying the configuration. Allow the initial ImageNet weight download. The official test split is required; selection uses official training counts only and excludes ambiguous file matches.
4. Review the plots, training curves and test reports. Section 10 exports `plant_model.pt` and `plant_model_meta.json` (optional ONNX). Section 12 downloads the two required files.
5. To demonstrate offline inference, copy both files and section 11's imports, loader, preprocessing and `predict` definition into a local Python session with compatible PyTorch, torchvision and Pillow already installed. Keep the model and JSON in the working directory; call `print(predict('my_leaf.jpg'))`. Do not copy the held-out-data plotting block unless the notebook data is available.

**Runtime:** download and GPU training duration are unmeasured; expect variability with archive connection, GPU allocation and number of retained images. Actual total runtime: [FILL]. Keep the session active and download outputs before disconnecting. No paid services are required by the code, but free Colab GPU access is not guaranteed.

## Fill before submission

- `report.docx`: names, roll numbers, submission date, genuine individual contributions, dataset/split counts, severity counts, mask-ratio comparison, parameter count, completed/best epochs, test accuracies and macro F1, TorchScript byte size and CPU timing. Add all five requested figure groups from the real notebook run, and update implementation status.
- Verify reference authors, years, titles and identifiers; add reliable sources for PD2SE-Net, apple black rot and Plantix. Confirm whether the instructor wants the separate university template front matter; this report deliberately follows the requested 1–9 structure instead.
- `viva_sheet.md`: replace result placeholders with observed values. Practice explaining the plots and limitations.
- Save the executed notebook with outputs and download the actual trained model and metadata for the demo.

## Verification and scope

Notebook code cells passed Python AST syntax checks. A temporary synthetic fixture with random images and masks exercised metadata matching, severity calculation, splits, loaders, untrained forward pass, evaluation, TorchScript export and CPU inference; temporary files were deleted. No optimization or real-data training was run in the preparation environment. The real archive layout, Colab GPU execution, pretrained weight download, ONNX export and real-data performance remain unverified. The optional EfficientNet comparison cell is omitted to keep the notebook focused and avoid an unvalidated comparison.

Severity is a whole-image mask-area proxy, not true leaf-area or expert severity: mild < 0.05, moderate from 0.05 to < 0.20, severe ≥ 0.20. The project is not a clinical or agricultural treatment recommendation, does not claim superiority, and is not a packaged mobile app.
