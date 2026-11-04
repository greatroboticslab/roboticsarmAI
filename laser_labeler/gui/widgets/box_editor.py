"""Edit one box's attributes in place: dropdowns from the taxonomy for object boxes, event and numbers for laser boxes."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core.models import INCIDENCES, LASER_KINDS, LaserBox, ObjectBox
from core.schema.class_names import COMPONENT_MEANING, COMPONENT_NAMES
from core.schema.taxonomy import Taxonomy


def _editable_combo() -> QComboBox:
    box = QComboBox()
    box.setEditable(True)
    box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    return box


class BoxEditor(QWidget):
    """Changes go straight into the dataclass passed to `set_box`, and `edited` is emitted."""
    edited = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._box: ObjectBox | LaserBox | None = None
        self._loading = False
        self._tax = Taxonomy()
        self._placeholder = "na"
        self.title = QLabel("Select a box on the image or in the list.")
        self.title.setWordWrap(True)
        # object fields
        self.general, self.sub, self.material, self.color = (_editable_combo() for _ in range(4))
        self.incidence = QComboBox()
        self.incidence.addItems(INCIDENCES)
        self.finish = QLineEdit()
        self.info = QLabel()
        self.info.setWordWrap(True)
        # laser fields
        self.kind = QComboBox()
        self.kind.addItems(LASER_KINDS)
        self.component = QComboBox()
        self.component.addItem("Not a part: class comes from the event", None)
        for name in COMPONENT_NAMES:
            self.component.addItem(f"{name}: {COMPONENT_MEANING[name]}", name)
        self.order = QSpinBox()
        self.order.setRange(0, 99)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Measurement", "Value (editable)"])
        self.metrics.horizontalHeader().setStretchLastSection(True)
        self.metrics.verticalHeader().hide()
        self.object_form_widget, self.laser_form_widget = QWidget(), QWidget()
        form = QFormLayout(self.object_form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        for label, widget in (("Object", self.general), ("Sub-object", self.sub), ("Material", self.material),
                              ("Color", self.color), ("Incidence", self.incidence), ("Finish", self.finish)):
            form.addRow(label, widget)
        lform = QFormLayout(self.laser_form_widget)
        lform.setContentsMargins(0, 0, 0, 0)
        lform.addRow("Laser event", self.kind)
        lform.addRow("Component", self.component)
        lform.addRow("Position along the beam", self.order)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        for w in (self.title, self.object_form_widget, self.laser_form_widget, self.info, self.metrics):
            col.addWidget(w)
        self.set_box(None)
        self.general.currentTextChanged.connect(self._general_changed)
        for combo, name in ((self.sub, "sub_object"), (self.material, "material"), (self.color, "color")):
            combo.currentTextChanged.connect(lambda text, n=name: self._set_attr(n, text))
        self.incidence.currentTextChanged.connect(lambda text: self._set_attr("incidence", text))
        self.finish.textEdited.connect(lambda text: self._set_attr("finish", text.strip() or None))
        self.kind.currentTextChanged.connect(lambda text: self._set_attr("kind", text))
        self.component.currentIndexChanged.connect(lambda _i: self._set_attr("component", self.component.currentData()))
        self.order.valueChanged.connect(lambda v: self._set_attr("order", int(v)))
        self.metrics.itemChanged.connect(self._metric_changed)

    # ------------------------------------------------------------------
    def set_taxonomy(self, taxonomy: Taxonomy, placeholder: str = "na") -> None:
        self._tax, self._placeholder = taxonomy, placeholder
        current = self._box
        self.set_box(current)

    def box(self):
        return self._box

    def set_box(self, box: ObjectBox | LaserBox | None) -> None:
        self._loading = True
        self._box = box
        is_object, is_laser = isinstance(box, ObjectBox), isinstance(box, LaserBox)
        self.object_form_widget.setVisible(is_object)
        self.laser_form_widget.setVisible(is_laser)
        self.metrics.setVisible(is_laser)
        self.info.setVisible(box is not None)
        if is_object:
            self.title.setText(f"<b>Object box {box.id}</b>")
            self._fill(self.general, list(self._tax.objects), box.general_object)
            self._fill(self.sub, self._tax.objects.get(box.general_object, []), box.sub_object)
            self._fill(self.material, self._tax.materials, box.material)
            self._fill(self.color, [*self._tax.colors, self._placeholder], box.color)
            self.incidence.setCurrentText(box.incidence)
            self.finish.setText(box.finish or "")
            self.info.setText(f"Confidence {box.confidence:.2f}. Hit by: {', '.join(box.laser_ids) or 'no laser box'}."
                              + (f"\nNote: {box.notes}" if box.notes else ""))
        elif is_laser:
            self.title.setText(f"<b>Laser box {box.id}</b>")
            self.kind.setCurrentText(box.kind)
            self.component.setCurrentIndex(max(self.component.findData(box.component), 0))
            self.order.setValue(box.order)
            self.info.setText((f"Part of {box.from_id}." if box.is_part else
                               f"Follows {box.from_id}." if box.from_id else "Starts the beam path.")
                              + (f"\nNote: {box.notes}" if box.notes else ""))
            self._fill_metrics(box)
        else:
            self.title.setText("Select a box on the image or in the list.")
        self._loading = False

    @staticmethod
    def _fill(combo: QComboBox, options: list[str], value: str) -> None:
        combo.clear()
        combo.addItems(options)
        combo.setCurrentText(value)  # a value outside the list stays visible, so the reviewer can see and fix it

    def _fill_metrics(self, box: LaserBox) -> None:
        self.metrics.blockSignals(True)
        self.metrics.setRowCount(len(box.metrics))
        for row, (key, value) in enumerate(sorted(box.metrics.items())):
            name = QTableWidgetItem(key)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metrics.setItem(row, 0, name)
            self.metrics.setItem(row, 1, QTableWidgetItem(f"{value:.6g}"))
        self.metrics.blockSignals(False)

    # ------------------------------------------------------------------
    def _set_attr(self, name: str, value) -> None:
        if self._loading or self._box is None or getattr(self._box, name, value) == value:
            return
        setattr(self._box, name, value)
        self.edited.emit()

    def _general_changed(self, text: str) -> None:
        if self._loading or not isinstance(self._box, ObjectBox):
            return
        if text != self._box.general_object:
            self._box.general_object = text
            subs = self._tax.objects.get(text)
            self._loading = True
            if subs is not None:  # a known object: offer its sub-objects and pick the first
                self._fill(self.sub, subs, subs[0])
                self._box.sub_object = subs[0]
            self._loading = False
            self.edited.emit()

    def _metric_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or not isinstance(self._box, LaserBox) or item.column() != 1:
            return
        key = self.metrics.item(item.row(), 0).text()
        try:
            value = float(item.text())
        except ValueError:
            self.metrics.blockSignals(True)
            item.setText(f"{self._box.metrics.get(key, 0):.6g}")  # not a number: put the old value back
            self.metrics.blockSignals(False)
            return
        if self._box.metrics.get(key) != value:
            self._box.metrics[key] = value
            self.edited.emit()
