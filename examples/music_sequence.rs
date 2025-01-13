//! This example demonstrates how to construct
//! looping music playback with an introduction.

use bevy::{log::LogPlugin, prelude::*};
use bevy_seedling::{
    music::{Label, MusicNode},
    SeedlingPlugin,
};
use firewheel::clock::ClockSeconds;

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins,
            LogPlugin::default(),
            AssetPlugin::default(),
            SeedlingPlugin::default(),
        ))
        .add_systems(Startup, play_music)
        .run();
}

fn play_music(server: Res<AssetServer>, mut commands: Commands) {
    // Here, we spawn a a sequence that has an introductory section,
    // a looping section, and a final "outro" section.

    let jump_label = Label(0);

    commands.spawn(
        MusicNode::build(server.load("midir.wav"))
            .label(jump_label, ClockSeconds(11.45))
            .jump(jump_label, ClockSeconds(70.094), ClockSeconds(0.25)),
    );
}
