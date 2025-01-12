#![expect(dead_code, missing_debug_implementations)]

use crate::sample::Sample;
use bevy_asset::prelude::*;
use bevy_ecs::prelude::*;
use core::ops::Range;
use firewheel::{
    clock::ClockSeconds,
    node::{AudioNode, AudioNodeProcessor, NodeEventType},
    param::{fixed_vec::FixedVec, AudioParam, ParamData, ParamEvent, PatchError},
    sample_resource::SampleResource,
    ChannelConfig, ChannelCount,
};
use std::sync::{
    atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
    Arc,
};

pub struct Loop {
    tail: Option<ClockSeconds>,
}

#[derive(Debug, Clone, Copy)]
pub struct Label(pub u32);

#[derive(Debug, Clone, Copy)]
pub enum SequenceData {
    Label(Label),
    Jump { target: Label, tail: ClockSeconds },
    ConditionalJump(Label),
}

impl SequenceData {
    pub fn is_label(&self, label: Label) -> bool {
        matches!(self, SequenceData::Label(l) if l.0 == label.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SequenceItem {
    timestamp: ClockSeconds,
    data: SequenceData,
}

#[derive(Debug, Clone, Copy)]
pub enum EventData {
    /// Fall through a loop point, continuing on in the sequence.
    Fallthrough,
    /// Jump to a labeled section.
    JumpTo(Label),
    /// Pause playback.
    Pause,
    /// Begin playing or resume playback.
    Play,
    /// Stop playback, resetting the playhead to the start.
    Stop,
    /// Set the playhead to the start and play the sequence.
    Restart,
}

#[derive(Debug, Clone, Copy)]
pub struct SequenceEvent {
    pub data: EventData,
    pub immediate: bool,
}

#[derive(Default, Clone)]
struct SequenceEvents {
    events: FixedVec<SequenceEvent>,
    consumed: usize,
    event_index: Arc<AtomicUsize>,
}

#[derive(Default, Debug)]
#[repr(u8)]
enum SequenceState {
    Playing,
    Paused,
    #[default]
    Stopped,
}

impl SequenceState {
    /// # Panics
    ///
    /// Panics if the integer is out of range.
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Playing,
            1 => Self::Paused,
            2 => Self::Stopped,
            _ => panic!("invalid integer for sequence state"),
        }
    }
}

#[derive(Clone)]
struct SequenceStateWrapper(Arc<AtomicU8>);

impl Default for SequenceStateWrapper {
    fn default() -> Self {
        Self(Arc::new(AtomicU8::new(SequenceState::default() as u8)))
    }
}

impl SequenceStateWrapper {
    pub fn new(state: SequenceState) -> Self {
        Self(Arc::new(AtomicU8::new(state as u8)))
    }

    pub fn get(&self) -> SequenceState {
        SequenceState::from_u8(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn set(&self, value: SequenceState) {
        self.0
            .store(value as u8, std::sync::atomic::Ordering::Relaxed)
    }
}

impl SequenceEvents {
    pub fn push(&mut self, event: SequenceEvent) {
        self.events.push(event);
        self.consumed += 1;
    }
}

impl firewheel::param::AudioParam for SequenceEvents {
    fn diff(
        &self,
        cmp: &Self,
        mut writer: impl FnMut(firewheel::param::ParamEvent),
        path: firewheel::param::ParamPath,
    ) {
        let newly_consumed = self.consumed.saturating_sub(cmp.consumed);

        if newly_consumed == 0 {
            return;
        }

        // If more items were added than the buffer can hold, we only have the most recent self.events.len() items.
        let clamped_newly_consumed = newly_consumed.min(self.events.len());

        // Start index for the new items. They are the last 'clamped_newly_consumed' items in the buffer.
        let start = self.events.len() - clamped_newly_consumed;
        let new_items = &self.events[start..];

        for event in new_items.iter() {
            writer(ParamEvent {
                data: ParamData::Any(Box::new(*event)),
                path: path.clone(),
            });
        }
    }

    fn patch(
        &mut self,
        data: &firewheel::param::ParamData,
        _: &[u32],
    ) -> Result<(), firewheel::param::PatchError> {
        match data.downcast_ref::<SequenceEvent>() {
            Some(event) => {
                self.push(*event);

                Ok(())
            }
            _ => Err(PatchError::InvalidData),
        }
    }
}

#[derive(Component)]
pub struct MusicNodeBuilder {
    music: Handle<Sample>,
    sequence: Vec<SequenceItem>,
    events: SequenceEvents,
}

pub(crate) fn insert_music_nodes(
    q: Query<(Entity, &MusicNodeBuilder)>,
    assets: Res<Assets<Sample>>,
    mut commands: Commands,
) {
    for (entity, builder) in q.iter() {
        if let Some(music) = assets.get(&builder.music) {
            commands
                .entity(entity)
                .remove::<MusicNodeBuilder>()
                .insert(MusicNode {
                    music: music.get(),
                    sequence: Arc::from(builder.sequence.as_slice()),
                    events: builder.events.clone(),
                    state: SequenceStateWrapper::new(SequenceState::Playing),
                    playhead: Default::default(),
                });
        }
    }
}

impl MusicNodeBuilder {
    pub fn label(mut self, label: Label, time: ClockSeconds) -> Self {
        self.sequence.push(SequenceItem {
            timestamp: time,
            data: SequenceData::Label(label),
        });

        self
    }

    pub fn jump(mut self, target: Label, time: ClockSeconds, tail: ClockSeconds) -> Self {
        self.sequence.push(SequenceItem {
            timestamp: time,
            data: SequenceData::Jump { target, tail },
        });

        self
    }

    pub fn condtional_jump(mut self, jump_to: Label, time: ClockSeconds) -> Self {
        self.sequence.push(SequenceItem {
            timestamp: time,
            data: SequenceData::ConditionalJump(jump_to),
        });

        self
    }
}

#[derive(Default, Clone)]
struct Playhead(Arc<AtomicU64>);

impl Playhead {
    fn set(&self, value: ClockSeconds) {
        let bits = value.0.to_bits();
        self.0.store(bits, Ordering::Relaxed);
    }

    fn get(&self) -> ClockSeconds {
        let bits = self.0.load(Ordering::Relaxed);

        ClockSeconds(f64::from_bits(bits))
    }
}

#[derive(crate::AudioParam, Clone, Component)]
pub struct MusicNode {
    music: Arc<dyn SampleResource>,
    sequence: Arc<[SequenceItem]>,
    events: SequenceEvents,
    #[param(skip)]
    state: SequenceStateWrapper,
    #[param(skip)]
    playhead: Playhead,
}

impl MusicNode {
    pub fn build(music: Handle<Sample>) -> MusicNodeBuilder {
        MusicNodeBuilder {
            music,
            sequence: Vec::new(),
            events: Default::default(),
        }
    }

    pub fn pause(&mut self) {
        self.events.push(SequenceEvent {
            data: EventData::Pause,
            immediate: false,
        });
    }

    pub fn play(&mut self) {
        self.events.push(SequenceEvent {
            data: EventData::Play,
            immediate: false,
        });
    }
}

impl From<MusicNode> for Box<dyn AudioNode> {
    fn from(value: MusicNode) -> Self {
        Box::new(value)
    }
}

impl AudioNode for MusicNode {
    fn debug_name(&self) -> &'static str {
        "low pass filter"
    }

    fn info(&self) -> firewheel::node::AudioNodeInfo {
        firewheel::node::AudioNodeInfo {
            num_min_supported_inputs: ChannelCount::ZERO,
            num_max_supported_inputs: ChannelCount::ZERO,
            num_min_supported_outputs: ChannelCount::MONO,
            num_max_supported_outputs: ChannelCount::STEREO,
            equal_num_ins_and_outs: false,
            default_channel_config: ChannelConfig {
                num_inputs: ChannelCount::ZERO,
                num_outputs: ChannelCount::STEREO,
            },
            updates: false,
            uses_events: true,
        }
    }

    fn activate(
        &mut self,
        stream_info: &firewheel::StreamInfo,
        _: ChannelConfig,
    ) -> Result<Box<dyn firewheel::node::AudioNodeProcessor>, Box<dyn std::error::Error>> {
        Ok(Box::new(MusicProcessor {
            params: self.clone(),
            sample_rate: stream_info.sample_rate.get() as f64,
            equal_power: EqualPower::new(8),
        }))
    }
}

struct EqualPower {
    a: Vec<f32>,
    b: Vec<f32>,
}

impl EqualPower {
    pub fn new(num_samples: usize) -> Self {
        let mut gains_a = Vec::with_capacity(num_samples);
        let mut gains_b = Vec::with_capacity(num_samples);

        // Compute the gain curves
        for i in 0..num_samples {
            let theta = std::f32::consts::FRAC_PI_2 * (i as f32 / (num_samples - 1) as f32);
            gains_a.push(theta.cos());
            gains_b.push(theta.sin());
        }

        Self {
            a: gains_a,
            b: gains_b,
        }
    }

    pub fn frames(&self) -> usize {
        self.a.len()
    }
}

pub struct MusicProcessor {
    params: MusicNode,
    sample_rate: f64,
    equal_power: EqualPower,
}

fn scan_for_event(range: Range<ClockSeconds>, events: &[SequenceItem]) -> Option<&SequenceItem> {
    let mut earliest: Option<&SequenceItem> = None;
    for item in events {
        if range.contains(&item.timestamp) {
            match earliest {
                Some(e) => {
                    if item.timestamp < e.timestamp {
                        earliest = Some(item);
                    }
                }
                None => {
                    earliest = Some(item);
                }
            }
        }
    }

    earliest
}

fn time_to_samples(time: ClockSeconds, rate: f64) -> u64 {
    (time.0 * rate).round() as u64
}

impl MusicProcessor {
    fn fill_buffers(
        &self,
        outputs: &mut [&mut [f32]],
        current_time: ClockSeconds,
        end_time: ClockSeconds,
    ) {
        let playhead = self.params.playhead.get();
        let start_frame = time_to_samples(playhead, self.sample_rate);

        let start_buff = time_to_samples(current_time, self.sample_rate) as usize;
        let end_buff = time_to_samples(end_time, self.sample_rate) as usize;

        self.params
            .music
            .fill_buffers(outputs, start_buff..end_buff, start_frame);
    }

    fn sample_range(&self, range: Range<ClockSeconds>) -> Range<ClockSeconds> {
        let playhead = self.params.playhead.get();
        range.start + playhead..range.end + playhead
    }

    fn item_delta(&self, item: &SequenceItem) -> ClockSeconds {
        let playhead = self.params.playhead.get();

        item.timestamp - playhead
    }

    fn next_event(&self, range: Range<ClockSeconds>) -> Option<&SequenceItem> {
        scan_for_event(self.sample_range(range), &self.params.sequence)
    }

    fn find_jump_target(&self, label: Label) -> Option<ClockSeconds> {
        for item in self.params.sequence.iter() {
            if item.data.is_label(label) {
                return Some(item.timestamp);
            }
        }

        None
    }
}

impl AudioNodeProcessor for MusicProcessor {
    fn process(
        &mut self,
        _: &[&[f32]],
        outputs: &mut [&mut [f32]],
        events: firewheel::node::NodeEventIter,
        proc_info: firewheel::node::ProcInfo,
    ) -> firewheel::node::ProcessStatus {
        for event in events {
            if let NodeEventType::Custom(event) = event {
                if let Some(param) = event.downcast_ref::<ParamEvent>() {
                    let _ = self.params.patch(&param.data, &param.path);
                }
            }
        }

        let state = self.params.state.get();
        if matches!(state, SequenceState::Paused | SequenceState::Stopped) {
            return firewheel::node::ProcessStatus::ClearAllOutputs;
        }

        let mut current_time = ClockSeconds(0.);
        let end_time =
            current_time + ClockSeconds(proc_info.frames as f64 * proc_info.sample_rate_recip);

        while current_time < end_time {
            let next_seq_event = self.next_event(current_time..end_time);

            match next_seq_event {
                Some(event) => {
                    self.fill_buffers(outputs, current_time, end_time);
                    current_time += self.item_delta(event);

                    match event.data {
                        SequenceData::Jump { target: jump, .. } => {
                            match self.find_jump_target(jump) {
                                Some(target) => {
                                    self.params.playhead.set(target);
                                }
                                None => {
                                    self.params.playhead.set(event.timestamp);
                                }
                            }
                        }
                        _ => {
                            self.params.playhead.set(event.timestamp);
                        }
                    }
                }
                None => {
                    self.fill_buffers(outputs, current_time, end_time);

                    let playhead = self.params.playhead.get();
                    self.params.playhead.set(end_time + playhead);

                    break;
                }
            }
        }

        for i in 0..proc_info.frames {
            outputs[0][i] *= 0.1;
            outputs[1][i] *= 0.1;
        }

        firewheel::node::ProcessStatus::outputs_not_silent()
    }
}
