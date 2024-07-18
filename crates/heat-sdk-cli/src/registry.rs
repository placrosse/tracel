#[derive(Clone, Debug)]
pub struct Flag {
    pub mod_path: &'static str,
    pub fn_name: &'static str,
    pub proc_type: &'static str,
    pub token_stream: &'static [u8],
}

impl Flag {
    pub fn new(mod_path: &'static str, fn_name: &'static str, proc_type: &'static str, token_stream: &'static [u8]) -> Self {
        Flag {
            mod_path,
            fn_name,
            proc_type,
            token_stream,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternCrate {
    pub imported_name: &'static str,
    pub importer_name: &'static str,
}

impl ExternCrate {
    pub fn new(imported_name: &'static str, importer_name: &'static str) -> Self {
        ExternCrate {
            imported_name,
            importer_name,
        }
    }
}

pub type LazyValue<T> = once_cell::sync::Lazy<T>;
pub struct Plugin<T: 'static>(pub &'static LazyValue<T>);

inventory::collect!(Plugin<Flag>);
inventory::collect!(Plugin<ExternCrate>);

pub const fn make_static_lazy<T: 'static>(func: fn() -> T) -> LazyValue<T> {
    LazyValue::<T>::new(func)
}

pub use gensym;
pub use inventory;
pub use paste;

// macro that generates a flag with a given type and arbitrary parameters and submits it to the inventory
#[macro_export]
macro_rules! register_flag {
    ($a:ty, $fn_:expr) => {
        $crate::registry::gensym::gensym! { $crate::register_flag!{ $a, $fn_ } }
    };
    ($gensym:ident, $a:ty, $fn_:expr) => {
        $crate::registry::paste::paste! {
            #[allow(non_upper_case_globals)]
            static [<$gensym _register_flag_>]: $crate::registry::LazyValue<$a> = $crate::registry::make_static_lazy(|| {
                $fn_
            });
            $crate::registry::inventory::submit!($crate::registry::Plugin(&[<$gensym _register_flag_>]));
        }
    };
}

pub(crate) fn get_flags() -> Vec<Flag> {
    inventory::iter::<Plugin<Flag>>
        .into_iter()
        .map(|plugin| (*plugin.0).to_owned())
        .collect()
}

pub(crate) fn get_external_crates() -> Vec<ExternCrate> {
    inventory::iter::<Plugin<ExternCrate>>
        .into_iter()
        .map(|plugin| (*plugin.0).to_owned())
        .collect()
}

/// The external crates flags consist of edges between the importer and the imported crates.
/// generate a visitable dependency tree from the flags in the goal of being able to know the paths to each imported crate transitively up to the root crate.
/// The root crate is the crate that has no importer.
/// 
/// Example: the root crate imports a crate A.
/// the flags will be :
/// ExternalCrate { imported_name: "A", importer_name: "root_crate" }
/// 
/// The dependency tree will be:
/// [
///     ["root_crate", "A"]
/// ]
/// 
/// Example 2: the root crate imports a crate A, and A imports B. And root crate also imports C.
/// the flags will be :
/// ExternalCrate { imported_name: "A", importer_name: "root_crate" }
/// ExternalCrate { imported_name: "B", importer_name: "A" }
/// ExternalCrate { imported_name: "C", importer_name: "root_crate" }
/// 
/// The dependency tree will be:
/// [
///     ["root_crate", "A", "B"],
///     ["root_crate", "C"]
/// ]
/// 
/// Example 3: the root crate imports a crate A, and A imports B. And root crate also imports C, and C imports B.
/// the flags will be :
/// ExternalCrate { imported_name: "A", importer_name: "root_crate" }
/// ExternalCrate { imported_name: "B", importer_name: "A" }
/// ExternalCrate { imported_name: "C", importer_name: "root_crate" }
/// ExternalCrate { imported_name: "B", importer_name: "C" }
/// 
/// The dependency tree will be:
/// [
///     ["root_crate", "A", "B"],
///     ["root_crate", "C", "B"]
/// ]
/// 
pub(crate) fn generate_dependency_tree(root_crate_name: &str) -> Vec<Vec<String>> {
    let mut dependency_tree = Vec::<Vec<String>>::new();
    let mut visited = std::collections::HashSet::<String>::new();
    let root_crate = root_crate_name.to_string();
    visited.insert(root_crate.clone());
    // Update stack to hold both crate name and its path
    let mut stack = Vec::<(String, Vec<String>)>::new();
    stack.push((root_crate.clone(), vec![root_crate.clone()]));

    while !stack.is_empty() {
        let (current_crate, current_path) = stack.pop().unwrap();
        let mut dependencies = Vec::<String>::new();
        for flag in get_external_crates() {
            if flag.importer_name == current_crate {
                dependencies.push(flag.imported_name.to_string());
            }
        }
        for dependency in dependencies {
            if !visited.contains(&dependency) {
                visited.insert(dependency.clone());
                let mut new_path = current_path.clone();
                new_path.push(dependency.clone());
                // Add the new path to the dependency tree
                dependency_tree.push(new_path.clone());
                // Push the dependency along with its path onto the stack
                stack.push((dependency, new_path));
            }
        }
    }
    dependency_tree
}